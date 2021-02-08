import torch


class Config(object):
    def __init__(self, vocab_size, embed_dim, label_num):
        self.model_name = 'TextLSTM'
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.label_num = label_num
        self.hidden_size = 128
        self.hidden_size = 50
        self.num_layer = 2
        self.dropout = 0.2
        self.lr = 0.001


class Model(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = torch.nn.Embedding(config.vocab_size, config.embed_dim)
        self.lstm = torch.nn.LSTM(config.embed_dim, config.hidden_size, config.label_num,
                                  dropout=config.dropout, bidirectional=True)
        self.fc = torch.nn.Linear(config.hidden_size * 2, config.label_num)

    def forward(self, input):
        embed = self.embedding(input)
        hidden, _ = self.lstm(embed)
        hidden = hidden[:, -1, :]
        logit = self.fc(hidden)
        return logit