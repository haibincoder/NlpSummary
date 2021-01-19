import math
import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, dim_model, dropout=0.1):
        super.__init__()


class Attention(nn.Module):
    def forward(self, query, key, value, mask=None, dropout=None):
        # 计算query key(转置)得到权重矩阵，除以dim(k)防止数据太大
        score = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(key.size(-1))
        # if mask:
        #     score = score.masked_fill(value == 0, -1e9)
        # dim=1对每一行进行softmax, dim=0对每一列进行softmax
        attention = torch.softmax(score, dim=1)

        if dropout:
            attention = torch.dropout(attention, dropout)
        z = torch.matmul(attention, value)
        return z, attention


class Layer(nn.Module):
    def __init__(self, features, eps=1e-6):
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class GLUE(nn.Module):
    '''
    use glue instead of relu
    '''
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((math.sqrt(2/math.pi) * (x * 0.044715 * torch.pow(x, 3)))))


if __name__ == "__main__":
    test = torch.tensor([[2.0, 5.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float)
    print(test.size())
    print(test.size(-1))
    print(test)
    print(test.transpose(-1, -2))
    #print('mask', test.masked_fill(, -1e9))
    softmax = torch.softmax(test, dim=1)
    softmax2 = torch.softmax(test, dim=0)
    print('softmax', softmax)
    print('softmax2', softmax2)