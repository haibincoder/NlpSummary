# coding: UTF-8
import torch
import torch.nn as nn
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, vocab_size, embed_dim, label_num):
        self.model_name = 'TextRNN'
        self.embedding_pretrained = None    # 预训练词向量
        self.dropout = 0.5
        self.num_classes = label_num                         # 类别数
        self.n_vocab = vocab_size                            # 词表大小，在运行时赋值
        self.batch_size = 128                                # mini-batch大小
        self.lr = 1e-3                                       # 学习率
        self.embed = embed_dim                               # 字向量维度, 若使用了预训练词向量，则维度统一
        self.hidden_size = 128                               # lstm隐藏层
        self.num_layers = 2                                  # lstm层数


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)  # [batch_size, seq_len, embeding]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out
