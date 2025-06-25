import numpy as np
import random
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig

from utils import to_gpu
from utils import ReverseLayerF


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)

def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)




class DeepSeekMoE(nn.Module):
    def __init__(self, input_size, output_size, dropout_rate=0.1, num_experts=8):
        super().__init__()
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        self.num_experts = num_experts

        # --- 多样性专家设计：每个专家结构不同 ---
        self.specialized_experts = nn.ModuleList()
        for i in range(num_experts):
            if i < num_experts // 4:  # 25% ReLU专家
                expert = nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.ReLU(),
                    nn.Linear(output_size, output_size)
                )
            elif i < num_experts // 2:  # 25% Tanh+BatchNorm专家
                expert = nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.Tanh(),
                    nn.BatchNorm1d(output_size),
                    nn.Linear(output_size, output_size)
                )
            elif i < 3 * num_experts // 4:  # 25% Conv1d+GELU专家
                expert = nn.Sequential(
                    nn.Unflatten(1, (1, input_size)),  # 调整维度适配Conv1d [B, 1, D]
                    nn.Conv1d(1, 4, kernel_size=3, padding=1),
                    nn.Flatten(1),  # [B, 4*D]
                    nn.Linear(4 * input_size, output_size),
                    nn.GELU()
                )
            else:  # 25% SiLU+LayerNorm专家
                expert = nn.Sequential(
                    nn.Linear(input_size, output_size),
                    nn.SiLU(),
                    nn.LayerNorm(output_size),
                    nn.Linear(output_size, output_size)
                )
            self.specialized_experts.append(expert)  # 修正append位置

        # --- 共享专家 ---
        self.shared_expert = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.LayerNorm(output_size),
            nn.Linear(output_size, output_size),
            nn.Dropout(dropout_rate)
        )

        # --- 模态感知门控网络 ---
        self.gate = nn.Sequential(
            nn.Linear(input_size + 1, input_size // 2),  # +1 for modality_id
            nn.Tanh(),
            nn.Linear(input_size // 2, num_experts)
        )

        # --- 可学习共享专家权重 ---
        self.shared_weight = nn.Parameter(torch.tensor(0.5))

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
            if module.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(module.weight)
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(module.bias, -bound, bound)

    def forward(self, x, modality_id=None):
        # modality_id: [batch_size], e.g., 0=text, 1=image, 2=audio
        batch_size = x.size(0)

        if modality_id is None:
            # 默认模态标记为 0（纯文本）
            modality_id = torch.zeros(batch_size, device=x.device)
        else:
            modality_id = modality_id.view(batch_size)  # [B]

        # --- 拼接模态标签用于门控 ---
        modality_onehot = modality_id.unsqueeze(1).float()  # [B, 1]
        gate_input = torch.cat([x, modality_onehot], dim=-1)  # [B, D+1]
        gate_logits = self.gate(gate_input)  # [B, num_experts]

        # --- 温度缩放 + 限幅稳定性 ---
        gate_logits = gate_logits / 0.7
        gate_logits = torch.clamp(gate_logits, -10, 10)

        # --- Top-k专家路由 ---
        k = min(4, self.num_experts)  # 动态选择k个专家
        topk_vals, topk_idx = torch.topk(gate_logits, k, dim=-1)
        topk_weights = F.softmax(topk_vals, dim=-1)  # [B, k]

        # --- 计算专家输出 ---
        expert_outputs = []
        for expert in self.specialized_experts:
            output = expert(x)
            # 确保所有专家输出维度一致
            if output.shape[1] != self.output_size:
                output = nn.functional.adaptive_avg_pool1d(output.unsqueeze(1), self.output_size).squeeze(1)
            expert_outputs.append(output)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, E, D]

        # --- 聚合Top-k专家输出（私有特征）---
        private_features = torch.zeros(batch_size, self.output_size, device=x.device)
        for i in range(k):
            idx = topk_idx[:, i]  # [B]
            weight = topk_weights[:, i]  # [B]
            expert_contribution = expert_outputs[torch.arange(batch_size), idx]  # [B, D]
            private_features += weight.unsqueeze(-1) * expert_contribution

        # --- 共享专家输出（共享特征）---
        shared_features = self.shared_expert(x)

        # --- 负载均衡损失 ---
        expert_usage = F.softmax(gate_logits, dim=-1).mean(0)  # [E]
        ideal_usage = torch.full_like(expert_usage, 1.0 / self.num_experts)  # [E]
        aux_loss = F.kl_div(torch.log(expert_usage + 1e-10), ideal_usage, reduction='batchmean')

        return shared_features, private_features, aux_loss
    

    
    


class DeepSHAPFusion(nn.Module):
    def __init__(self, feature_dim, output_dim, num_modalities=3):
        super().__init__()
        self.num_modalities = num_modalities
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, output_dim)
        )
        
        # 门控网络学习权重
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, num_modalities),
            nn.Sigmoid()  # 输出(0,1)区间的权重
        )
        
        # 噪声分支（GMF模块的一部分）
        self.noise_branch = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2),
            nn.ReLU(),
            nn.Linear(feature_dim//2, feature_dim)
        )

    def forward(self, features):
        B, M, D = features.shape
        
        # 1. 计算SHAP模态贡献度
        base_output = self.projection(features.mean(dim=1)).detach()
        shap_deltas = []
        for i in range(M):
            keep_mask = torch.arange(M, device=features.device) != i
            partial_fused = features[:, keep_mask, :].mean(dim=1)
            partial_output = self.projection(partial_fused)
            delta = torch.abs(base_output - partial_output).mean(dim=1, keepdim=True)
            shap_deltas.append(delta)
            
        shap_deltas = torch.cat(shap_deltas, dim=1)  # [B, M]
        contrib_scores = shap_deltas  # 模态贡献分数
        
        # 2. 门控网络学习权重（DMC模块）
        global_repr = features.mean(dim=1)  # 全局表示 [B, D]
        learned_weights = self.gate_network(global_repr)  # [B, M]
        
        # 3. 模态感知融合（GMF模块）
        # 加权特征
        weighted_features = features * learned_weights.unsqueeze(-1)  # [B, M, D]
        
        # 带噪声的融合（GMF模块的一部分）
        noise = self.noise_branch(global_repr).unsqueeze(1)  # [B, 1, D]
        fused_features = torch.cat([weighted_features, noise], dim=1)  # [B, M+1, D]
        
        # 最终融合
        fused = fused_features.mean(dim=1)  # [B, D]
        final_output = self.projection(fused)
        
        return final_output, learned_weights, contrib_scores
    
    # def dcl_rank_loss(self, learned_weights, contrib_scores):
    #     """计算排序一致性损失"""
    #     # 获取排序索引， 新增：温度缩放提高梯度灵敏度
    #     pred_rank = torch.argsort(learned_weights*5, dim=1, descending=True)
    #     true_rank = torch.argsort(contrib_scores*5, dim=1, descending=True)
        
    #     # 将排序索引转换为排名
    #     pred_ranking = torch.zeros_like(pred_rank)
    #     true_ranking = torch.zeros_like(true_rank)
        
    #     for i in range(pred_rank.shape[0]):
    #         for j in range(pred_rank.shape[1]):
    #             pred_ranking[i, pred_rank[i, j]] = j
    #             true_ranking[i, true_rank[i, j]] = j
        
    #     # 计算排名差异的L1损失
    #     loss = torch.abs(pred_ranking.float() - true_ranking.float()).mean()
    #     return loss
    
    # def boost(self, learned_weights, eta=0.2):
    #     """促进模态权重稀疏性的损失"""
    #     dynamic_eta = torch.clamp(eta - torch.mean(learned_weights, dim=0), min=0.05)
    #     return torch.relu(dynamic_eta - learned_weights).sum(dim=1).mean()

    def normalize_contrib(contrib_scores):
        """
        对贡献值进行 batch-wise min-max 归一化
        contrib_scores: Tensor，形状 [B, M]，表示每个样本每个模态的贡献值
        返回归一化后的贡献值，范围是0~1
        """
        min_c = contrib_scores.min(dim=1, keepdim=True)[0]  # 每个样本的最小贡献值
        max_c = contrib_scores.max(dim=1, keepdim=True)[0]  # 每个样本的最大贡献值
        normed = (contrib_scores - min_c) / (max_c - min_c + 1e-6)  # 防止除零
        return normed



    def dcl_rank_loss(self,pred_weights, contrib_scores):
        """
        简化版排序监督损失
        输入：
        pred_weights: [B, M] 模型预测的模态融合权重（未归一化）
        contrib_scores: [B, M] SHAP计算出的模态贡献值（未归一化）

        输出：
        loss，标量
        """

        # softmax 得到权重概率分布
        pred_prob = F.softmax(pred_weights, dim=1)
        contrib_prob = F.softmax(contrib_scores, dim=1)

        # 计算均值
        pred_mean = pred_prob.mean(dim=1, keepdim=True)
        contrib_mean = contrib_prob.mean(dim=1, keepdim=True)

        # 中心化
        pred_centered = pred_prob - pred_mean
        contrib_centered = contrib_prob - contrib_mean

        # 计算 Pearson 相关系数分子和分母
        numerator = (pred_centered * contrib_centered).sum(dim=1)
        denominator = torch.sqrt((pred_centered ** 2).sum(dim=1)) * torch.sqrt((contrib_centered ** 2).sum(dim=1)) + 1e-8

        corr = numerator / denominator  # 相关系数，越接近1越好

        loss = 1 - corr.mean()  # 相关性越高，loss 越小

        return loss
    

    def boost_loss(self,weights, eta=0.15):
        """
        双向惩罚 BoostLoss，鼓励模态权重均衡
        weights: [B, M] 经过 softmax 后的权重
        eta: 门限，推荐 0.3 左右

        返回：
        标量 loss
        """
        loss_low = F.relu(eta - weights)        # 低于 eta 的部分惩罚
        loss_high = F.relu(weights - (1 - eta)) # 高于 1-eta 的部分惩罚

        loss = (loss_low + loss_high).sum(dim=1).mean()
        return loss

    
    



    




# let's define a simple model that can deal with multimodal variable length sequence
class MISA_CMDC(nn.Module):
    def __init__(self, config):
        super(MISA_CMDC, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()


        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
        self.trnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_t = nn.Sequential()
        self.project_t.add_module('project_t',
                                  nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.hidden_size))
        self.project_t.add_module('project_t_activation', self.activation)
        self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v',
                                  nn.Linear(in_features=hidden_sizes[1] * 4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[2] * 4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())




        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 6,
                                                           out_features=self.config.hidden_size * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.config.hidden_size * 3, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        # 原来是，但会出现警告
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths.to('cpu'))

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.to('cpu'))

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):

        batch_size = lengths.size(0)


        # 从文本模态中提取特征
        final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
        utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video, utterance_audio)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator(
            (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a) / 3.0)

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t,
                         self.utt_shared_v, self.utt_shared_a), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        o = self.fusion(h)
        shared_embs = torch.cat((self.utt_shared_t, self.utt_shared_a, self.utt_shared_v), 1)
        diff_embs = torch.cat((self.utt_private_t, self.utt_private_a, self.utt_private_v), 1)

        return o, shared_embs, diff_embs

    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a):

        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        self.utt_shared_a = self.shared(utterance_a)

    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)
        o, shared_embs, diff_embs = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o, shared_embs, diff_embs


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        #基本初始化
        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.visual_size
        self.acoustic_size = config.acoustic_size


        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        
        
        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU

        #为每个模态设计初步特征提取器
        if self.config.use_bert:

            # Initializing a BERT bert-base-uncased style configuration
            try:
                bertconfig = BertConfig.from_pretrained('./bert-base-uncased/', output_hidden_states=True)
                self.bertmodel = BertModel.from_pretrained('./bert-base-uncased/', config=bertconfig)
            except:
                bertconfig = BertConfig.from_pretrained('../bert-base-uncased/', output_hidden_states=True)
                self.bertmodel = BertModel.from_pretrained('../bert-base-uncased/', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2*hidden_sizes[0], hidden_sizes[0], bidirectional=True)
        
        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2*hidden_sizes[1], hidden_sizes[1], bidirectional=True)
        
        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2*hidden_sizes[2], hidden_sizes[2], bidirectional=True)



        ##########################################
        # 不同模态的特征映射到同一维度空间，方便特征对齐
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=hidden_sizes[0]*4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=hidden_sizes[1]*4, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a', nn.Linear(in_features=hidden_sizes[2]*4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))


         ##########################################
        # deepseekmoe变量定义
        ##########################################

        self.deepseekmoe = DeepSeekMoE(input_size=config.hidden_size, output_size=config.hidden_size)


        ##########################################
        # 新增：MI 引导模块，用于私有特征引导共享特征
        ##########################################
        self.mi_t = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.mi_v = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        self.mi_a = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.ReLU(),
            nn.Linear(config.hidden_size, config.hidden_size)
        )

        ##########################################
        # DeepSHAP融合模块
        ##########################################
        self.deepshap_fusion = DeepSHAPFusion(
            feature_dim=config.hidden_size,  # 输入维度
            output_dim=config.output_dim     # 输出维度
        )




        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))



        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2', nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1', nn.Linear(in_features=config.hidden_size, out_features=4))


        ##########################################
        # 特征融合模块
        ##########################################
        # self.fusion = nn.Sequential()
        # self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size*6, out_features=self.config.hidden_size*3))
        # self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        # self.fusion.add_module('fusion_layer_1_activation', self.activation)
        # self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.config.hidden_size*3, out_features= output_size))
        #self.fusion1 = nn.Sequential()
        #self.fusion1.add_module('fusion_layer', nn.Linear(in_features=self.config.hidden_size, out_features= output_size))
        
        self.gate_linear=nn.Linear(in_features=6, out_features= 6)

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0]*2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1]*2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2]*2,))

        # 原来是，但会出现警告
        # encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        

        
    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths.to('cpu'))

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.to('cpu'))

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, sentences, visual, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        
        batch_size = lengths.size(0)
        
        if self.config.use_bert:
            bert_output = self.bertmodel(input_ids=bert_sent, 
                                         attention_mask=bert_sent_mask, 
                                         token_type_ids=bert_sent_type)      

            bert_output = bert_output[0]

            # masked mean
            masked_output = torch.mul(bert_sent_mask.unsqueeze(2), bert_output)
            mask_len = torch.sum(bert_sent_mask, dim=1, keepdim=True)  
            bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len
            utterance_text = bert_output
        else:
            # extract features from text modality
            sentences = self.embed(sentences)
            final_h1t, final_h2t = self.extract_features(sentences, lengths, self.trnn1, self.trnn2, self.tlayer_norm)
            utterance_text = torch.cat((final_h1t, final_h2t), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
            

        # extract features from visual modality
        final_h1v, final_h2v = self.extract_features(visual, lengths, self.vrnn1, self.vrnn2, self.vlayer_norm)
        utterance_video = torch.cat((final_h1v, final_h2v), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)

        # extract features from acoustic modality
        final_h1a, final_h2a = self.extract_features(acoustic, lengths, self.arnn1, self.arnn2, self.alayer_norm)
        utterance_audio = torch.cat((final_h1a, final_h2a), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        
        
        # Projecting to same sized space
        self.utt_t_orig = utterance_text = self.project_t(utterance_text)
        self.utt_v_orig = utterance_video = self.project_v(utterance_video)
        self.utt_a_orig = utterance_audio = self.project_a(utterance_audio)
        
        #使用deepseekmoe
        self.utt_shared_t, self.utt_private_t, aux_t = self.deepseekmoe(utterance_text)
        self.utt_shared_v, self.utt_private_v, aux_v = self.deepseekmoe(utterance_video)
        self.utt_shared_a, self.utt_private_a, aux_a = self.deepseekmoe(utterance_audio)

        # self.utt_shared_t, self.utt_private_t = self.deepseekmoe(utterance_text)
        # self.utt_shared_v, self.utt_private_v= self.deepseekmoe(utterance_video)
        # self.utt_shared_a, self.utt_private_a = self.deepseekmoe(utterance_audio)

        # Shared-private encoders
        # self.shared_private(utterance_text, utterance_video, utterance_audio)


        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None


        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator( (self.utt_shared_t + self.utt_shared_v + self.utt_shared_a)/3.0 )
        
        # For reconstruction
        self.reconstruct()
        
        # 1-LAYER TRANSFORMER FUSION
        # h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_private_a, self.utt_shared_t, self.utt_shared_v,  self.utt_shared_a), dim=0)
        # h = self.transformer_encoder(h)
        # h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        # o = self.fusion(h)
        
        
        # Step 1-2: 私有特征引导共享特征（MI 结构）
        c_t = self.mi_t(self.utt_private_t + self.utt_shared_t)  # 文本模态
        c_v = self.mi_v(self.utt_private_v + self.utt_shared_v)  # 视觉模态
        c_a = self.mi_a(self.utt_private_a + self.utt_shared_a)  # 音频模态

         # Step 3: 准备特征用于DeepSHAP融合
        features = torch.stack([c_t, c_v, c_a], dim=1)  # [B, 3, hidden_size]

        # # Step 4: 使用DeepSHAPFusion计算最终预测和模态权重
        # o, modality_weights, contrib_scores= self.deepshap_fusion(features)


        # rank_loss = self.deepshap_fusion.dcl_rank_loss(modality_weights, contrib_scores)
        # boost_loss = self.deepshap_fusion.boost(modality_weights)
        
        # Step 4: 使用DeepSHAPFusion计算最终预测和模态权重
        o, learned_weights, contrib_scores= self.deepshap_fusion(features)


        rank_loss = self.deepshap_fusion.dcl_rank_loss(learned_weights, contrib_scores)
        boost_loss = self.deepshap_fusion.boost_loss(learned_weights)

        # rank_loss = rank_loss * 2.0  # 放大梯度，增强影响
        # boost_loss = boost_loss * 5.0


   

        #加了负载均衡
        shared_embs = torch.cat((self.utt_shared_t, self.utt_shared_a, self.utt_shared_v), 1)
        diff_embs = torch.cat((self.utt_private_t, self.utt_private_a, self.utt_private_v), 1)
        aux_loss =  aux_t + aux_v + aux_a
        aux_loss = aux_loss / aux_loss.max() 
        # aux_loss = aux_loss * 2.0

        # return o, shared_embs, diff_embs, aux_loss

        return o, shared_embs, diff_embs, aux_loss, rank_loss, boost_loss
        # return o, shared_embs, diff_embs, rank_loss,boost_loss
    
    def reconstruct(self,):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        self.utt_a_recon = self.recon_a(self.utt_a)


    # def shared_private(self, utterance_t, utterance_v, utterance_a):

    #     # Private-shared components
    #     self.utt_private_t = self.private_t(utterance_t)
    #     self.utt_private_v = self.private_v(utterance_v)
    #     self.utt_private_a = self.private_a(utterance_a)

    #     self.utt_shared_t = self.shared(utterance_t)
    #     self.utt_shared_v = self.shared(utterance_v)
    #     self.utt_shared_a = self.shared(utterance_a)


    def forward(self, sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask):
        batch_size = lengths.size(0)

        o, shared_embs, diff_embs,aux_loss, rank_loss,boost_loss = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        return o, shared_embs, diff_embs,aux_loss, rank_loss,boost_loss

        # o, shared_embs, diff_embs, rank_loss,boost_loss = self.alignment(sentences, video, acoustic, lengths, bert_sent, bert_sent_type, bert_sent_mask)
        # return o, shared_embs, diff_embs, rank_loss,boost_loss

