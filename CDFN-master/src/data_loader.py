import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer
from create_dataset import MOSI, MOSEI, UR_FUNNY, PAD, UNK, CMDC, IEMOCAP

# config = BertConfig.from_json_file("../bert-base-uncased/config.json")
# model = BertModel.from_pretrained("../bert-base-uncased/pytorch_model.bin", config=config)
try:
    bert_tokenizer = BertTokenizer.from_pretrained('./bert-base-uncased/')  # 注意此处为本地文件夹
except:
    bert_tokenizer = BertTokenizer.from_pretrained('../bert-base-uncased/')  # 注意此处为本地文件夹

class MSADataset(Dataset):
    def __init__(self, config):

        ## Fetch dataset
        if "mosi" in str(config.data_dir).lower():
            dataset = MOSI(config)  #处理MOSI数据集,  dataset变量存储self.train，self.dev，self.test，self.pretrained_emb, self.word2id
        elif "mosei" in str(config.data_dir).lower():
            dataset = MOSEI(config)
        elif "ur_funny" in str(config.data_dir).lower():
            dataset = UR_FUNNY(config)
        elif "cmdc" in str(config.data_dir).lower():
            dataset = CMDC(config)
        elif "iemocap" in str(config.data_dir).lower():
            dataset = IEMOCAP(config)
        else:
            print("Dataset not defined correctly")
            exit()

        self.data, self.word2id, self.pretrained_emb = dataset.get_data(config.mode) #根据 config.mode 指定的模式获取相应的数据
        self.len = len(self.data)

        config.visual_size = self.data[0][0][1].shape[1]
        config.acoustic_size = self.data[0][0][2].shape[1]

        config.word2id = self.word2id
        config.pretrained_emb = self.pretrained_emb


    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len



def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""
    #dataset存放的是划分好的数据集、预训练的嵌入矩阵、单词到索引的映射
    dataset = MSADataset(config) #根据配置信息选择合适的数据集类，通过dataset变量来访问和操作该数据集。

    print(config.mode)
    config.data_len = len(dataset)


    def collate_fn(batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        # for later use we sort the batch in descending order of length
        batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

        # get the data out of the batch - use pad sequence util functions from PyTorch to pad things


        labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
        try:
            if (batch[0][0][0].shape[1] == 768 and batch[0][0][1].shape[1] == 768 and batch[0][0][2].shape[1] == 128) or \
                    (batch[0][0][0].shape[1] == 300 and batch[0][0][1].shape[1] == 35 and batch[0][0][2].shape[1] == 74):
                # 其实限定了是CMDC或者是iemocap, 直接提好了TXT特征
                sentences = pad_sequence([torch.FloatTensor(sample[0][0]) for sample in batch])
        except:
            sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
        visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])


        ## BERT-based features input prep

        SENT_LEN = sentences.size(0)
        # Create bert indices using tokenizer
        try:
            bert_details = []
            for sample in batch:
                text = " ".join(sample[0][3])

                # 原版警告
                # encoded_bert_sent = bert_tokenizer.encode_plus(
                #     text, max_length=SENT_LEN+2, add_special_tokens=True, pad_to_max_length=True)

                encoded_bert_sent = bert_tokenizer.encode_plus(
                    text, max_length=SENT_LEN + 2, add_special_tokens=True, truncation=True, padding='max_length')
                bert_details.append(encoded_bert_sent)

            # Bert things are batch_first
            bert_sentences = torch.LongTensor([sample["input_ids"] for sample in bert_details])
            bert_sentence_types = torch.LongTensor([sample["token_type_ids"] for sample in bert_details])
            bert_sentence_att_mask = torch.LongTensor([sample["attention_mask"] for sample in bert_details])

        except: # 将V1 版本中提取的特征直接sum好赋值过来即可 使用随机mask会出错,acc为1, loss为nan!
            if batch[0][0][0].shape[1] == 300 and batch[0][0][1].shape[1] == 35 and batch[0][0][2].shape[1] == 74:
                bert_sentences = None
                bert_sentence_types = None
                bert_sentence_att_mask = None
            else:
                labels = labels.float() # UR_FUNNY 要将label也变为 'torch.FloatTensor'
                sum_sentences = torch.sum(sentences, dim=0).float() # ([64, 32, 300]) -> [32, 300] 消除第一个特征维度 64 保留对齐的300和32 bathsize 维度
                bert_sentences = (sum_sentences-torch.min(sum_sentences))/(torch.max(sum_sentences)-torch.min(sum_sentences)) # 将矩阵归一化到 0-1 之间 [32, 300]
                # 生成全一矩阵
                shape  = (bert_sentences.size(0), bert_sentences.size(1))
                bert_sentence_types = torch.ones(shape)
                bert_sentence_att_mask = torch.ones(shape)

        # lengths are useful later in using RNNs
        lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])

        return sentences, visual, acoustic, labels, lengths, bert_sentences, bert_sentence_types, bert_sentence_att_mask


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
