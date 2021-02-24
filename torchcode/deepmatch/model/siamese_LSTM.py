import torch
import torch.nn.functional as F

"""
利用孪生LSTM计算文本相似度
"""

batch_size = 10
vocab_size = 100
seq_length = 3
embedding_dim = 200



# 2个label, 相似/不相似
lstm_hidden_size = 2

class SiameseLSTM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, batch_size, lstm_hidden_size, bidirectional=False)
        self.fc = torch.nn.Linear(5, 2)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, input_1, input_2=None, labels=None):
        # 输入batch_size * seq_length, 输出 batch_size * seq_length * embedding_dim
        embedding1 = self.embedding(input_1)
        if input_2 is None:
            output1, (hn, cn) = self.lstm(embedding1)
            return output1[:, -1, :]
        embedding2 = self.embedding(input_2)

        # 输入 batch_size * seq_length * embedding_dim， 输出output: batch_size * seq_length * hidden_size
        output1, (hn, cn) = self.lstm(embedding1)
        output2, (hn, cn) = self.lstm(embedding2)
        logit1 = output1[:, -1, :]
        logit2 = output2[:, -1, :]
        loss = self.loss_fn(logit1, logit2)

        if labels is not None:
            logits = F.cosine_similarity(logit1, logit2)
            distance_matrix = 1 - logits   # [1, 0, 1, 0, 0 , 1]
            negs = distance_matrix[labels == 0]
            poss = distance_matrix[labels == 1]

            # 选择最难识别的正负样本
            negative_pairs = negs[negs < (
                poss.max() if len(poss) > 1 else negs.mean())]
            positive_pairs = poss[poss > (
                negs.min() if len(negs) > 1 else poss.mean())]

            positive_loss = positive_pairs.pow(2).sum()
            negative_loss = F.relu(1 - negative_pairs).pow(2).sum()
            loss = positive_loss + negative_loss

        return loss


if __name__ == "__main__":
    # 输入 batch_size * seq_length
    input1 = torch.randint(0, vocab_size, (batch_size, seq_length))
    input2 = torch.randint(0, vocab_size, (batch_size, seq_length))

    labels = torch.randint(0, 2, (1, batch_size)).squeeze(0)

    model = SiameseLSTM()
    loss = model(input1, input2, labels)
    print(loss)
    hidden = model(input1)
    print(hidden)

