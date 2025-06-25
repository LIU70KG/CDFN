import os
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from scipy.stats import pearsonr
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
from thop import profile, clever_format
from create_dataset import CMDC_PHQ9_labels
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
import models
from shutil import copyfile, rmtree


class Solver(object):
    def __init__(self, train_config, dev_config, test_config, train_data_loader, dev_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.dev_data_loader = dev_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    

    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config) 
        
        # Final list
        for name, param in self.model.named_parameters():

            # Bert freezing customizations 
            if self.train_config.data == "mosei":
                if "bertmodel.encoder.layer" in name:
                    layer_num = int(name.split("encoder.layer.")[-1].split(".")[0])
                    if layer_num <= (8):
                        param.requires_grad = False
            elif self.train_config.data == "ur_funny":
                if "bert" in name:
                    param.requires_grad = False
            
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)

        # Initialize weight of Embedding matrix with Glove embeddings
        if not self.train_config.use_bert and (self.train_config.data != 'cmdc' and self.train_config.data != 'CMDC'):
            if self.train_config.pretrained_emb is not None:
                self.model.embed.weight.data = self.train_config.pretrained_emb
            self.model.embed.requires_grad = False
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)


    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        criterion_MAE = nn.L1Loss(reduction="mean")

        # self.criterion = criterion = nn.L1Loss(reduction="mean")
        if self.train_config.data == "ur_funny":
            self.criterion = criterion = nn.CrossEntropyLoss(reduction="mean")
        elif self.train_config.data == 'iemocap':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else: # mosi and mosei are regression datasets
            self.criterion = criterion = nn.MSELoss(reduction="mean")

        self.domain_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.sp_loss_criterion = nn.CrossEntropyLoss(reduction="mean")
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()  # Similarity Loss
        
        # best_valid_loss = float('inf')
        best_mae, best_rmse, best_pearsonrn = float('inf'), float('inf'), float('-inf')
        best_precision, best_recall, best_f1, best_accuracy, best_multiclass_acc= 0.0, 0.0, 0.0, 0.0, 0.0
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)
        continue_epochs = 0
        
        # # 加载最优模型
        # if os.path.isfile('src/bestpoints/model_2025-06-24_21:27:36.std'):
        #     print("Loading pretrained weights...")
        #     self.model.load_state_dict(torch.load(
        #         f'src/bestpoints/model_2025-06-24_21:27:36.std'))
        
        #     self.optimizer.load_state_dict(torch.load(
        #         f'src/bestpoints/optim_2025-06-24_21:27:36.std'))
            
        #     mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(mode="test")
        #     print('_test_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
        #     print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))
        #     continue_epochs = 36
        # print("continue iter:", continue_epochs)

        # 初始化用于存储训练和测试 MAE 的列表
        train_mae_history = []
        test_mae_history = []
        # 设置绘图窗口
        # plt.ion()  # 开启交互模式
        # fig, ax = plt.subplots(figsize=(10, 6))

        checkpoints = 'src/checkpoints'
        if os.path.exists(checkpoints):
                rmtree(checkpoints, ignore_errors=False, onerror=None)
        os.makedirs(checkpoints)

        for e in range(continue_epochs, self.train_config.n_epoch):
            print(f"-----------------------------------epoch{e}---------------------------------------")
            print(f"//Current patience: {curr_patience}, current trial: {num_trials}.//")
            self.model.train()

            train_loss_cls, train_loss_sim, train_loss_diff = [], [], []
            # train_loss_recon = []
            train_loss_sp = []
            train_loss = []
            y_true, y_pred = [], []
            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                batch_size = t.size(0)
                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                try:
                    bert_sent = to_gpu(bert_sent)
                    bert_sent_type = to_gpu(bert_sent_type)
                    bert_sent_mask = to_gpu(bert_sent_mask)
                except:
                    pass
                
                #y_tilde, shared_embs, diff_embs= self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                # y_tilde, shared_embs, diff_embs,rank_loss,boost_loss = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                


                #加了负载均衡之后
                y_tilde, shared_embs, diff_embs, aux_loss, rank_loss, boost_loss= self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                
                
                # # # 使用 thop 计算 FLOPs 和参数数量
                # if self.train_config.data == "cmdc":
                #     flops, params = profile(self.model, inputs=(t[:,:1,:], v[:,:1,:], a[:,:1,:], l[:1], bert_sent[:1,:], bert_sent_type[:1,:], bert_sent_mask[:1,:]))
                #     flops, params = clever_format([flops, params], "%.3f")
                #     print(f"Train_FLOPs: {flops}")
                #     print(f"Train_Parameters: {params}")
                #
                # if self.train_config.data == "mosi":
                #     flops, params = profile(self.model, inputs=(t[:,1:2], v[:,1:2,:], a[:,1:2,:], l[1:2], bert_sent[1:2,:], bert_sent_type[1:2,:], bert_sent_mask[1:2,:]))
                #     flops, params = clever_format([flops, params], "%.3f")
                #     print(f"Train_FLOPs: {flops}")
                #     print(f"Train_Parameters: {params}")

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                if self.train_config.data == 'iemocap':
                    y_tilde = y_tilde.view(-1, 2)
                    cls_loss = (self.criterion(y_tilde[::4], y[::4]).mean() + self.criterion(y_tilde[1::4], y[1::4]).mean() + \
                    self.criterion(y_tilde[2::4], y[2::4]).mean() + self.criterion(y_tilde[3::4], y[3::4]).mean())/4
                else:
                    cls_loss = criterion(y_tilde, y)
                diff_loss = self.get_diff_loss()
                domain_loss = self.get_domain_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()
                
                if self.train_config.use_cmd_sim:
                    similarity_loss = cmd_loss
                else:
                    similarity_loss = domain_loss
                


                #加了负载均衡后总损失
                loss = 2 * cls_loss + \
                    self.train_config.diff_weight * diff_loss + \
                    self.train_config.sim_weight * similarity_loss + \
                    self.train_config.recon_weight * recon_loss + \
                    self.train_config.aux_loss_weight * aux_loss + \
                    self.train_config.rank_weight * rank_loss+ \
                    self.train_config.boost_weight * boost_loss
                # 加了负载均衡后总损失
                # loss = cls_loss + \
                #     self.train_config.diff_weight * diff_loss + \
                #     self.train_config.sim_weight * similarity_loss + \
                #     self.train_config.aux_loss_weight * aux_loss + \
                #     self.train_config.rank_weight * rank_loss+ \
                #     self.train_config.boost_weight * boost_loss


                # loss = cls_loss + \
                #     self.train_config.diff_weight * diff_loss + \
                #     self.train_config.sim_weight * similarity_loss + \
                #     self.train_config.recon_weight * recon_loss + \
                #     self.train_config.rank_weight * rank_loss+ \
                #     self.train_config.boost_weight * boost_loss

                loss.backward()

                
                torch.nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad], self.train_config.clip)
                self.optimizer.step()

                train_loss.append(loss.item())
                train_loss_cls.append(cls_loss.item())
                train_loss_diff.append(diff_loss.item())
                train_loss_sim.append(similarity_loss.item())
                # train_loss_recon.append(recon_loss.item())
                # 记录样本的预测值和真实值，用于计算指标
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())
                

            print(f"Training loss: {round(np.mean(train_loss), 4)}")
            # print('train_loss_cls:%.4f./ train_loss_diff:%.4f./ train_loss_sim:%.4f./ train_loss_recon:%.4f./' % (
            print('train_loss_cls:%.4f./ train_loss_diff:%.4f./ train_loss_sim:%.4f./ ' % (
            round(np.mean(train_loss_cls), 4), round(np.mean(train_loss_diff), 4),round(np.mean(train_loss_sim), 4)))
            round(np.mean(train_loss_sim), 4)
            # round(np.mean(train_loss_recon), 4)))
            print(f"--------------------------------------------")
            # valid_loss, valid_acc = self.eval(mode="dev", epoch=e)
            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()
            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.calc_metrics(y_true, y_pred)
            print('_train_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
            print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.eval(e, mode="test")
            print('_test_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
            print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

            print(f"Current patience: {curr_patience}, current trial: {num_trials}.")
            flag = 0
            if best_mae > mae:
                best_mae = mae
                rmse_bestmae = rmse
                pearsonrn_bestmae = pearsonrn
                precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae = precision, recall, f1, accuracy, multiclass_acc
                flag = 1
            if best_rmse > rmse:
                best_rmse = rmse
                flag = 1
            if best_accuracy < accuracy:
                best_accuracy = accuracy
                # flag = 1
            if best_multiclass_acc < multiclass_acc:
                best_multiclass_acc = multiclass_acc
                # flag = 1
            if best_f1 < f1:
                best_precision, best_recall, best_f1 = precision, recall, f1
                # flag = 1
            if flag == 1:
                print("------------------Found new best model on test set!----------------")
                print(f"epoch: {e}")
                print("mae: ", mae)
                print("rmse: ", rmse)
                print("Pearsonrn/Corr: ", pearsonrn)
                print("precision: ", precision)
                print("recall: ", recall)
                print("f1: ", f1)
                print("accuracy: ", accuracy)
                print("multiclass_acc: ", multiclass_acc)
                if not os.path.exists('src/checkpoints'): os.makedirs('src/checkpoints')
                torch.save(self.model.state_dict(), f'src/checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'src/checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'src/checkpoints/model_{self.train_config.name}.std', weights_only=True))
                    self.optimizer.load_state_dict(torch.load(f'src/checkpoints/optim_{self.train_config.name}.std', weights_only=True))
                    lr_scheduler.step()
                    print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")

            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break

        # self.eval(mode="test", to_print=True)
        print("------------------best all on test set----------------")
        print('_best_mae:%.4f. / best_rmse:%.4f. / best_f1:%.4f. / best_accuracy: %.4f. / best_multiclass_acc: %.4f.' % (best_mae, best_rmse, best_f1, best_accuracy, best_multiclass_acc))
        print("------------------best MAE on test set----------------")
        mae, rmse, pearsonrn = best_mae, rmse_bestmae, pearsonrn_bestmae
        precision, recall, f1, accuracy, multiclass_acc= precision_bestmae, recall_bestmae, f1_bestmae, accuracy_bestmae, multiclass_acc_bestmae
        print('_test_MAE:%.4f.   RMSE:%.4f.  Pearsonrn/Corr:%.4f.' % (mae, rmse, pearsonrn))
        print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./ multiclass_acc:%.4f./' % (precision, recall, f1, accuracy, multiclass_acc))

        # 判断文件是否存在
        if  not os.path.exists(self.train_config.best_model_Configuration_Log):
            # 如果文件存在，则清空文件
            with open(self.train_config.best_model_Configuration_Log, 'w') as f:
                pass  
                
        with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
            F1.write("\n" + "="*180 + "\n")  # 分隔线，长度可调
            line = 'sim_weight:{sim_weight} | diff_weight:{diff_weight} |aux_loss_weight:{aux_loss_weight}|rank_weight:{rank_weight} |boost_weight:{boost_weight} |activation:{activation} |learning_rate:{learning_rate} | \n ' \
                   'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \
                   'precision:{precision} | recall:{recall} | f1:{f1} | accuracy:{accuracy} | multiclass_acc:{multiclass_acc} |\n' \
                   'best_mae:{best_mae} | best_rmse:{best_rmse} | best_f1:{best_f1} | best_accuracy:{best_accuracy} | best_multiclass_acc:{best_multiclass_acc} |\n' \
                .format(sim_weight=self.train_config.sim_weight,
                        diff_weight=self.train_config.diff_weight,
                        aux_loss_weight=self.train_config.aux_loss_weight,
                        rank_weight=self.train_config.rank_weight,
                        boost_weight=self.train_config.boost_weight,
                        activation=self.train_config.activation,
                        learning_rate=self.train_config.learning_rate,
                        test_MAE=mae,
                        RMSE=rmse,
                        Pearson=pearsonrn,
                        precision=precision,
                        recall=recall,
                        f1=f1,
                        accuracy=accuracy,
                        multiclass_acc=multiclass_acc,
                        best_mae=best_mae,
                        best_rmse=best_rmse,
                        best_f1=best_f1,
                        best_accuracy=best_accuracy,
                        best_multiclass_acc=best_multiclass_acc,
                        )

            print('result saved～')
            F1.write(line)

        # with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
        #     F1.write("\n" + "="*180 + "\n")  # 分隔线，长度可调
        #     line = 'sim_weight:{sim_weight} | diff_weight:{diff_weight} | recon_weight:{recon_weight} |aux_loss_weight:{aux_loss_weight} |activation:{activation} |learning_rate:{learning_rate} | \n ' \
        #            'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \
        #            'precision:{precision} | recall:{recall} | f1:{f1} | accuracy:{accuracy} | multiclass_acc:{multiclass_acc} |\n' \
        #            'best_mae:{best_mae} | best_rmse:{best_rmse} | best_f1:{best_f1} | best_accuracy:{best_accuracy} | best_multiclass_acc:{best_multiclass_acc} |\n' \
        #         .format(sim_weight=self.train_config.sim_weight,
        #                 diff_weight=self.train_config.diff_weight,
        #                 recon_weight=self.train_config.recon_weight,
        #                 aux_loss_weight=self.train_config.aux_loss_weight,
        #                 activation=self.train_config.activation,
        #                 learning_rate=self.train_config.learning_rate,
        #                 test_MAE=mae,
        #                 RMSE=rmse,
        #                 Pearson=pearsonrn,
        #                 precision=precision,
        #                 recall=recall,
        #                 f1=f1,
        #                 accuracy=accuracy,
        #                 multiclass_acc=multiclass_acc,
        #                 best_mae=best_mae,
        #                 best_rmse=best_rmse,
        #                 best_f1=best_f1,
        #                 best_accuracy=best_accuracy,
        #                 best_multiclass_acc=best_multiclass_acc,
        #                 )

        #     print('result saved～')
        #     F1.write(line)

        # with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
        #     F1.write("\n" + "="*180 + "\n")  # 分隔线，长度可调
        #     line = 'sim_weight:{sim_weight} | diff_weight:{diff_weight} | _weight:{recon_weighrecont} |aux_loss_weight:{aux_loss_weight}|rank_weight:{rank_weight} |boost_weight:{boost_weight}|activation:{activation} |learning_rate:{learning_rate} | \n ' \
        #            'test_best_MAE:-----------{test_MAE}------------ | RMSE:{RMSE} | Pearson:{Pearson} |\n' \
        #            'precision:{precision} | recall:{recall} | f1:{f1} | accuracy:{accuracy} | multiclass_acc:{multiclass_acc} |\n' \
        #            'best_mae:{best_mae} | best_rmse:{best_rmse} | best_f1:{best_f1} | best_accuracy:{best_accuracy} | best_multiclass_acc:{best_multiclass_acc} |\n' \
        #         .format(sim_weight=self.train_config.sim_weight,
        #                 diff_weight=self.train_config.diff_weight,
        #                 recon_weight=self.train_config.recon_weight,
        #                 aux_loss_weight=self.train_config.aux_loss_weight,
        #                 rank_weight=self.train_config.rank_weight,
        #                 boost_weight=self.train_config.boost_weight,
        #                 activation=self.train_config.activation,
        #                 learning_rate=self.train_config.learning_rate,
        #                 test_MAE=mae,
        #                 RMSE=rmse,
        #                 Pearson=pearsonrn,
        #                 precision=precision,
        #                 recall=recall,
        #                 f1=f1,
        #                 accuracy=accuracy,
        #                 multiclass_acc=multiclass_acc,
        #                 best_mae=best_mae,
        #                 best_rmse=best_rmse,
        #                 best_f1=best_f1,
        #                 best_accuracy=best_accuracy,
        #                 best_multiclass_acc=best_multiclass_acc,
        #                 )

        #     print('result saved～')
        #     F1.write(line)

        return mae
    
    
    #测试模型
    def eval(self, e, mode=None, to_print=False, best=False):
        assert(mode is not None)
        self.model.eval()

        y_true, y_pred = [], []
        eval_loss, eval_loss_diff = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            self.model.load_state_dict(torch.load(
                f'src/checkpoints/model_{self.train_config.name}.std'))

        features, labels = [], []
        with torch.no_grad():

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                l = to_gpu(l)
                try:
                    bert_sent = to_gpu(bert_sent)
                    bert_sent_type = to_gpu(bert_sent_type)
                    bert_sent_mask = to_gpu(bert_sent_mask)
                except:
                    pass

                #y_tilde, shared_embs, diff_embs = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                # y_tilde, shared_embs, diff_embs,rank_loss,boost_loss = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)


                #加入负载均衡
                y_tilde, shared_embs, diff_embs, aux_loss,rank_loss,boost_loss = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask)
                
                
                # # # 使用 thop 计算 FLOPs 和参数数量
                # if self.train_config.data == "cmdc":
                #     flops, params = profile(self.model, inputs=(t[:,:1,:], v[:,:1,:], a[:,:1,:], l[:1], bert_sent[:1,:], bert_sent_type[:1,:], bert_sent_mask[:1,:]))
                #     flops, params = clever_format([flops, params], "%.3f")
                #     print(f"Test_FLOPs: {flops}")
                #     print(f"Test_Parameters: {params}")
                #
                # if self.train_config.data == "mosi":
                #     flops, params = profile(self.model, inputs=(t[:,:1], v[:,:1,:], a[:,:1,:], l[:1], bert_sent[:1,:], bert_sent_type[:1,:], bert_sent_mask[:1,:]))
                #     flops, params = clever_format([flops, params], "%.3f")
                #     print(f"Test_FLOPs: {flops}")
                #     print(f"Test_Parameters: {params}")

                if self.train_config.data == "ur_funny":
                    y = y.squeeze()
                if self.train_config.data == 'iemocap':
                    y_tilde = y_tilde.view(-1, 2)
                    cls_loss = (self.criterion(y_tilde[::4], y[::4]).mean() + self.criterion(y_tilde[1::4], y[1::4]).mean() + \
                    self.criterion(y_tilde[2::4], y[2::4]).mean() + self.criterion(y_tilde[3::4], y[3::4]).mean())/4
                else:
                    cls_loss = self.criterion(y_tilde, y)
                loss = cls_loss
                eval_loss.append(loss.item())
                y_pred.append(y_tilde.detach().cpu().numpy())
                y_true.append(y.detach().cpu().numpy())

        #         features.append(shared_embs[:, :128].contiguous().view(-1, self.train_config.hidden_size).cpu().numpy())
        #         features.append(shared_embs[:, 128:256].contiguous().view(-1, self.train_config.hidden_size).cpu().numpy())
        #         features.append(shared_embs[:, 256:384].contiguous().view(-1, self.train_config.hidden_size).cpu().numpy())
        #         features.append(diff_embs[:, :128].contiguous().view(-1, self.train_config.hidden_size).cpu().numpy())
        #         features.append(diff_embs[:, 128:256].contiguous().view(-1, self.train_config.hidden_size).cpu().numpy())
        #         features.append(diff_embs[:, 256:384].contiguous().view(-1, self.train_config.hidden_size).cpu().numpy())
        #         labels.append(np.ones(shape=(shared_embs.shape[0])) * 0)
        #         labels.append(np.ones(shape=(shared_embs.shape[0])) * 1)
        #         labels.append(np.ones(shape=(shared_embs.shape[0])) * 2)
        #         labels.append(np.ones(shape=(diff_embs.shape[0])) * 3)
        #         labels.append(np.ones(shape=(diff_embs.shape[0])) * 4)
        #         labels.append(np.ones(shape=(diff_embs.shape[0])) * 5)


        # # # # tsne
        # # if to_print or epoch == 500:
        # if 1==1:
        #     from sklearn.manifold import TSNE
        #     import matplotlib.pyplot as plt
        #     import pandas as pd
        #     expression = ['text-shared', 'audio-shared', 'vision-shared', 'text-private', 'audio-private', 'vision-private']
        #     colors = ['deepskyblue', 'green', 'blue', 'red', 'orange', 'pink']
        
        #     features1 = np.concatenate(features, axis=0)
        #     labels1 = np.concatenate(labels, axis=0)
        #     print(features1.shape, labels1.shape)
        
        #     tsne1 = TSNE(n_components=2, init="pca", random_state=1, perplexity=30)
        #     x_tsne1 = tsne1.fit_transform(features1)
        #     print("Data has the {} before tSNE and the following after tSNE {}".format(features1.shape[-1],
        #                                                                                x_tsne1.shape[-1]))
        #     x_min, x_max = x_tsne1.min(0), x_tsne1.max(0)
        #     X_norm1 = (x_tsne1 - x_min) / (x_max - x_min)
        
        #     ''' plot results of tSNE '''
        #     fake_df1 = pd.DataFrame(X_norm1, columns=['X', 'Y'])
        #     fake_df1['Group'] = labels1
        
        #     group_codes1 = {k: colors[idx] for idx, k in enumerate(fake_df1.Group.unique())}
        #     fake_df1['colors'] = fake_df1['Group'].apply(lambda x: group_codes1[x])
        
        #     # 将像素值转换为英寸
        #     width_in_inches = 1196 / 100
        #     height_in_inches = 802 / 100
        #     # 创建固定大小的图像
        #     fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=100)
        
        #     for i in range(6):
        #         ax.scatter(X_norm1[fake_df1['Group'] == i, 0], X_norm1[fake_df1['Group'] == i, 1], c=group_codes1[i],
        #                    label=expression[i], s=40, marker='o', linewidths=1)
        #     # plt.title('Decomposed features', fontsize=15)
        
        #     #		plt.legend(loc = 1, fontsize = 'small')
        #     # ax.legend(fontsize=30, bbox_to_anchor=(-0.015, 0.98, 0.1, 0.1), loc='lower left', ncol=3, columnspacing=1)
        #     ax.legend(fontsize=24, bbox_to_anchor=(-0.07, 0.98, 0.1, 0.1), loc='lower left', ncol=3,
        #                       columnspacing=0.2, handletextpad=0.05)

        #     plt.savefig(f'src/figure/mosi_{e}.png', bbox_inches='tight')
        #     plt.close("all")

        eval_loss = np.mean(eval_loss)
        y_true = np.concatenate(y_true, axis=0).squeeze()
        y_pred = np.concatenate(y_pred, axis=0).squeeze()

        if self.train_config.data == 'iemocap':
            test_preds = y_pred.reshape((y_pred.shape[0] // 4, 4, 2))
            test_truth = y_true.reshape(-1, 4)
            f1, acc = [], []
            for emo_ind in range(4):
                test_preds_i = np.argmax(test_preds[:, emo_ind], axis=1)
                test_truth_i = test_truth[:, emo_ind]
                f1.append(f1_score(test_truth_i, test_preds_i, average='weighted'))
                acc.append(accuracy_score(test_truth_i, test_preds_i))

            accuracy = AverageAcc = np.mean(acc)
            Averagef1 = np.mean(f1)
            if to_print:
                ne, ha, sa, an = acc
                ne_f, ha_f, sa_f, an_f = f1
                print('HappyAcc:%.4f.  SadAcc:%.4f.   AngryAcc:%.4f.   NeutralAcc:%.4f. AverageAcc:%.4f.' % (
                ha, sa, an, ne, AverageAcc))
                print('HappyF1:%.4f.  SadF1:%.4f.   AngryF1:%.4f.   NeutralF1:%.4f. Averagef1:%.4f.' % (
                ha_f, sa_f, an_f, ne_f, Averagef1))

        else:
            mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc = self.calc_metrics(y_true, y_pred,
                                                                                                      mode, to_print)
            return mae, rmse, pearsonrn, precision, recall, f1, accuracy, multiclass_acc

        return eval_loss, accuracy
    
    #计算多分类任务的准确率
    def multiclass_acc(self, preds, truths):
        """
        Compute the multiclass accuracy w.r.t. groundtruth
        :param preds: Float array representing the predictions, dimension (N,)
        :param truths: Float/int array representing the groundtruth classes, dimension (N,)
        :return: Classification accuracy
        """
        return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))
    
    
    def calc_metrics(self, y_true, y_pred, mode=None, to_print=False):
        if self.train_config.data == "ur_funny":
            test_preds = np.argmax(y_pred, 1)
            test_truth = y_true

            if to_print:
                print("Confusion Matrix (pos/neg) :")
                print(confusion_matrix(test_truth, test_preds))
                print("Classification Report (pos/neg) :")
                print(classification_report(test_truth, test_preds, digits=5))
                print("Accuracy (pos/neg) ", accuracy_score(test_truth, test_preds))
                print("这里要改一下，不用这个数据集将不要改")
                exit()
            return accuracy_score(test_truth, test_preds)
        
        elif self.train_config.data == "cmdc":
            test_preds = y_pred
            test_truth = y_true
            mae = np.mean(np.absolute(test_preds - test_truth))
            rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))
            pearsonrn, p_value = pearsonr(test_preds, test_truth)
            binary_preds = test_preds >= 5
            binary_truth = test_truth >= 5
            precision = precision_score(binary_truth, binary_preds, zero_division=1)
            recall = recall_score(binary_truth, binary_preds, zero_division=1)
            f1 = f1_score(binary_truth, binary_preds)
            mult_a2=accuracy_score(binary_truth, binary_preds)
            multiclass_true = np.array(CMDC_PHQ9_labels(y_true))
            multiclass_pred = np.array(CMDC_PHQ9_labels(y_pred))
            multiclass_acc = np.sum(multiclass_true == multiclass_pred) / float(len(multiclass_pred))
            return mae, rmse, pearsonrn, precision, recall, f1, mult_a2, multiclass_acc

        else:
            test_preds = y_pred
            test_truth = y_true

            non_zeros = np.array([i for i, e in enumerate(test_truth) if e != 0])

            test_preds_a7 = np.clip(test_preds, a_min=-3., a_max=3.)
            test_truth_a7 = np.clip(test_truth, a_min=-3., a_max=3.)

            mae = np.mean(np.absolute(test_preds - test_truth))   # Average L1 distance between preds and truths
            rmse = np.sqrt(np.mean((test_preds - test_truth) ** 2))

            if np.all(test_preds == test_preds[0]) or np.all(test_truth == test_truth[0]):
                corr = 0
                print("Warning: One of the input arrays is constant; correlation is undefined.")
            else:
                corr = np.corrcoef(test_preds, test_truth)[0, 1]

            mult_a7 = self.multiclass_acc(test_preds_a7, test_truth_a7)
            binary_truth = (test_truth >= 0)
            binary_preds = (test_preds >= 0)
            precision = precision_score(binary_truth, binary_preds, zero_division=1)
            recall = recall_score(binary_truth, binary_preds, zero_division=1)
            f1 = f1_score(binary_truth, binary_preds)
            mult_a2=accuracy_score(binary_truth, binary_preds)

            return mae, rmse, corr, precision, recall, f1, mult_a2, mult_a7



    def get_domain_loss(self,):

        if self.train_config.use_cmd_sim:
            return 0.0
        
        # Predicted domain labels
        domain_pred_t = self.model.domain_label_t
        domain_pred_v = self.model.domain_label_v
        domain_pred_a = self.model.domain_label_a

        # True domain labels
        domain_true_t = to_gpu(torch.LongTensor([0]*domain_pred_t.size(0)))
        domain_true_v = to_gpu(torch.LongTensor([1]*domain_pred_v.size(0)))
        domain_true_a = to_gpu(torch.LongTensor([2]*domain_pred_a.size(0)))

        # Stack up predictions and true labels
        domain_pred = torch.cat((domain_pred_t, domain_pred_v, domain_pred_a), dim=0)
        domain_true = torch.cat((domain_true_t, domain_true_v, domain_true_a), dim=0)

        return self.domain_loss_criterion(domain_pred, domain_true)

    def get_cmd_loss(self,):

        if not self.train_config.use_cmd_sim:
            return 0.0

        # losses between shared states
        loss = self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_v, 5)
        loss += self.loss_cmd(self.model.utt_shared_t, self.model.utt_shared_a, 5)
        loss += self.loss_cmd(self.model.utt_shared_a, self.model.utt_shared_v, 5)
        loss = loss/3.0

        return loss

    def get_diff_loss(self):

        shared_t = self.model.utt_shared_t
        shared_v = self.model.utt_shared_v
        shared_a = self.model.utt_shared_a
        private_t = self.model.utt_private_t
        private_v = self.model.utt_private_v
        private_a = self.model.utt_private_a

        # Between private and shared
        loss = self.loss_diff(private_t, shared_t)
        loss += self.loss_diff(private_v, shared_v)
        loss += self.loss_diff(private_a, shared_a)

        # Across privates
        loss += self.loss_diff(private_a, private_t)
        loss += self.loss_diff(private_a, private_v)
        loss += self.loss_diff(private_t, private_v)

        return loss
    
    def get_recon_loss(self, ):

        loss = self.loss_recon(self.model.utt_t_recon, self.model.utt_t_orig)
        loss += self.loss_recon(self.model.utt_v_recon, self.model.utt_v_orig)
        loss += self.loss_recon(self.model.utt_a_recon, self.model.utt_a_orig)
        loss = loss/3.0
        return loss





