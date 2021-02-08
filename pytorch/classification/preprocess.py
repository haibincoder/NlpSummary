

if __name__ == "__main__":
    path = '../../data/THUCNews/data/train.txt'
    input_data = open(path, 'r', encoding='utf-8')
    train_data = set()
    for item in input_data.readlines():
        item_str, _ = item.split('\t')
        for item_char in item_str:
            train_data.add(item_char)

    vocab = {'<PAD>': 0, '<NUL>': 1}
    index = 2
    for item in train_data:
        vocab[item] = index
        index += 1
    output = open('../../data/THUCNews/data/vocab.txt', 'w', encoding='utf-8')
    for key, value in enumerate(vocab):
        output.writelines(f'{value}\t{key}\n')
    print('build vocab success')
