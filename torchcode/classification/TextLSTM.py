import torch
from torch import nn


class Config(object):
    def __init__(self, vocab_size, embed_dim, label_num):
        self.model_name = 'TextLSTM'
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.label_num = label_num
        self.hidden_size = 128
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)
        self.lstm = nn.LSTM(config.embed_dim, config.hidden_size, config.num_layer,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc = nn.Linear(config.hidden_size * 2, config.label_num)

    def forward(self, input):
        # input: batchsize,seq_length = 128, 50
        embed = self.embedding(input)
        # embed: batchsize,seq_length,embed_dim = 128, 50, 300
        hidden, _ = self.lstm(embed)
        # hidden: batchsize, seq, embedding = 128, 50, 256
        hidden = hidden[:, -1, :]
        # hidden: batchsize, seq_embedding = 128, 256
        logit = torch.sigmoid(self.fc(hidden))
        # logit: batchsize, label_logit = 128, 10
        return logit