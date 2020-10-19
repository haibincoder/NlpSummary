import random
import pandas as pd

if __name__ == "__main__":
    df = pd.read_excel('../data/train.xlsx', dtype=str)
    result = []

    for item in df.itertuples():
        result_item = {}
        result_item['用户问'] = item[0]
        result_item['答案'] = item[1]
        result_item['标注结果'] = ''
        result.append(result_item)

    # 随机采样
    random.shuffle(result)
    if len(result) > 1000:
        result = result[:1000]

    output = pd.DataFrame(result)
    output.to_excel('../out/output.xlsx')