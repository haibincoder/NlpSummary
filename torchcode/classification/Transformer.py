import copy

from torch import nn
import torch.nn.functional as F
import torch
import math


class Config(object):
    """配置参数"""

    def __init__(self, vocab_size, embed_dim, label_num, max_length=32):
        self.embedding_pretrained = None
        self.num_classes = label_num  # 类别数
        self.vocab_size = vocab_size  # 词表大小，在运行时赋值
        self.embed_dim = embed_dim  # 字向量维度
        self.num_head = 5
        self.dropout = 0.1
        self.hidden = 512
        self.num_encoder = 2
        self.max_length = max_length
        self.lr = 1e-3


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)

        self.encoder = Encoder(config.embed_dim, config.num_head, config.hidden, config.dropout)
        self.encoders = nn.ModuleList([
            copy.deepcopy(self.encoder)
            for _ in range(config.num_encoder)])

        self.fc1 = nn.Linear(config.max_length * config.embed_dim, config.num_classes)

    def forward(self, x):
        out = self.embedding(x)
        for encoder in self.encoders:
            out = encoder(out)
        out = out.view(out.size(0), -1)
        # out = torch.mean(out, 1)
        out = self.fc1(out)
        return out


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_head, hidden, dropout=0.0):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_head, dropout)
        self.feed_forward = Position_wise_Feed_Forward(embed_dim, hidden, dropout)

    def forward(self, x):
        out = self.attention(x)
        out = self.feed_forward(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_head, dropout=0.0):
        super().__init__()
        self.num_head = num_head
        assert embed_dim % num_head == 0, 'head num error'
        self.dim_head = embed_dim // num_head
        self.fc_q = nn.Linear(embed_dim, embed_dim)
        self.fc_k = nn.Linear(embed_dim, embed_dim)
        self.fc_v = nn.Linear(embed_dim, embed_dim)
        self.attention = Attention()
        self.fc = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        Q = Q.view(batch_size * self.num_head, -1, self.dim_head)
        K = K.view(batch_size * self.num_head, -1, self.dim_head)
        V = V.view(batch_size * self.num_head, -1, self.dim_head)
        context = self.attention(Q, K, V)
        context = context.view(batch_size, -1, self.dim_head * self.num_head)
        out = self.fc(context)
        out = self.dropout(out)
        # 残差
        out = out + x
        out = self.layer_norm(out)
        return out


class Attention(nn.Module):
    """
    Attention计算
    """
    def __init__(self):
        super().__init__()

    def forward(self, query, key, value):
        k = query.size(-1)
        result = torch.matmul(query, key.transpose(-2, -1))
        score = result / math.sqrt(k)
        softmax_result = torch.softmax(score, dim=-1)
        result = torch.matmul(softmax_result, value)
        return result


'''
这里对应transformers encoder的Feed forward
'''


class Position_wise_Feed_Forward(nn.Module):
    def __init__(self, dim_model, hidden, dropout=0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim_model, hidden)
        self.fc2 = nn.Linear(hidden, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = out + x  # 残差连接
        out = self.layer_norm(out)
        return out


if __name__ == "__main__":
    test = torch.tensor([[2.0, 5.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float)
    print(test.size())
    print(test.size(-1))
    print(test)
    print(test.transpose(-1, -2))
    # print('mask', test.masked_fill(, -1e9))
    softmax = torch.softmax(test, dim=1)
    softmax2 = torch.softmax(test, dim=0)
    print('softmax', softmax)
    print('softmax2', softmax2)

    input = torch.tensor([[2.0, 5.0, 3.0], [1.0, 2.0, 3.0]])
    layer = nn.LayerNorm([2, 3])
    norm = layer(input)
    print('input', input)
    print('m', norm)
