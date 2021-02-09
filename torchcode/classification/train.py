from importlib import import_module

import torch
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

vocab_size = 5000
batch_size = 128
max_length = 32
embed_dim = 300
label_num = 10
epoch = 5
train_path = '../../data/THUCNews/data/train.txt'
dev_path = '../../data/THUCNews/data/dev.txt'
vocab_path = '../../data/THUCNews/data/vocab.txt'


def get_data(path):
    input_vocab = open(vocab_path, 'r', encoding='utf-8')
    vocabs = {}
    for item in input_vocab.readlines():
        word, wordid = item.replace('\n', '').split('\t')
        vocabs[word] = int(wordid)
    input_data = open(path, 'r', encoding='utf-8')
    x = []
    y = []
    for item in input_data.readlines():
        sen, label = item.replace('\n', '').split('\t')
        tmp = []
        for item_char in sen:
            if item_char in vocabs:
                tmp.append(vocabs[item_char])
            else:
                tmp.append(1)
            if len(tmp) >= max_length:
                break
        x.append(tmp)
        y.append(int(label))

    # padding
    for item in x:
        if len(item) < max_length:
            item += [0] * (max_length - len(item))

    label_num = len(set(y))
    # x_train, x_test, y_train, y_test = train_test_split(np.array(x), np.array(y), test_size=0.2)
    x = np.array(x)
    print(x.shape)
    y = np.array(y)
    return x, y, label_num


class DealDataset(Dataset):
    def __init__(self, x_train, y_train, device):
        self.x_data = torch.from_numpy(x_train).long().to(device)
        self.y_data = torch.from_numpy(y_train).long().to(device)
        self.len = x_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def evaluate(model, dataloader_dev):
    model.eval()
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for datas, labels in dataloader_dev:
            output = model(datas)
            predic = torch.max(output.data, 1)[1].cpu()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, labels)
            if len(predict_all) > 1000:
                break
    acc = metrics.accuracy_score(labels_all, predict_all)
    return acc


if __name__ == "__main__":
    debug = False
    module = import_module('TextLstmMax')
    config = module.Config(vocab_size, embed_dim, label_num)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = module.Model(config).to(device)
    if debug:
        inputs = torch.randint(0, 200, (batch_size, embed_dim))
        labels = torch.randint(0, 2, (batch_size, 1)).squeeze(0)
        print(model(inputs))
    else:
        x_train, y_train, label_num = get_data(train_path)
        dataset = DealDataset(x_train, y_train, device)
        dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

        x_dev, y_dev, _ = get_data(dev_path)
        dataset_dev = DealDataset(x_dev, y_dev, device)
        dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        model.train()
        for i in range(epoch):
            index = 0
            for datas, labels in tqdm(dataloader):
                model.zero_grad()
                output = model(datas)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                optimizer.step()
                index += 1
                if index % 50 == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(output.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc = evaluate(model, dataloader_dev)
                    print(f'epoch:{i} item:{index} loss:{loss} train_acc:{train_acc} dev_acc:{dev_acc}')
                    model.train()

        print('train finish')