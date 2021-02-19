# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, vocab_size, embed_dim, label_num):
        self.model_name = 'TextCNN'
        self.embedding_pretrained = None
        self.dropout = 0.2
        self.num_classes = label_num                                    # 类别数
        self.n_vocab = vocab_size                                       # 词表大小，在运行时赋值
        self.embed_dim = embed_dim                                          # 字向量维度
        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 256                                          # 卷积核数量(channels数)
        self.lr = 1e-3


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed_dim, padding_idx=config.n_vocab - 1)
        # self.convs = nn.ModuleList(
        #     [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.conv1 = nn.Conv2d(1, config.num_filters, (2, config.embed_dim))
        self.conv2 = nn.Conv2d(1, config.num_filters, (3, config.embed_dim))
        self.conv3 = nn.Conv2d(1, config.num_filters, (4, config.embed_dim))
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input):
        out = self.embedding(input)
        out = out.unsqueeze(1)
        # out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        conv1 = self.conv_and_pool(out, self.conv1)
        conv2 = self.conv_and_pool(out, self.conv2)
        conv3 = self.conv_and_pool(out, self.conv3)
        out = torch.cat((conv1, conv2, conv3), 1)

        out = self.dropout(out)
        out = self.fc(out)
        return out
