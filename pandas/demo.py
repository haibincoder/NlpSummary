import pandas as pd


def apply_func_demo(x: int) -> int:
    return x*10


def apply_func_concat(x: pd.DataFrame) -> str:
    if x['age'] % 2 == 0:
        return '男'
    else:
        return '女'


if __name__=="__main__":
    column = ['name', 'age']
    df = pd.DataFrame(columns=column)

    # 插入行
    for index in range(0, 10):
        obj = {}
        obj['name'] = f'test_{index}'
        obj['age'] = index
        df_new = pd.DataFrame(obj, index=[0])
        df = df.append(df_new, ignore_index=True)

    # 插入列
    df['sex'] = ['0' for i in range(0, 10)]

    # dataframe执行函数
    df['count'] = df['age'].astype(int).apply(apply_func_demo)
    df['sex'] = df.apply(apply_func_concat, axis=1)
    # axis=0为每一行，axis=1为每一列
    df = df.apply(lambda x: x*1000 if x.name in ['age'] else x, axis=0)

    # dataframe删除列
    # df.drop('age', axis=1)
    # 删除行
    # df.drop(0, axis=0)

    print(f'行数:{df.shape[0]}')
    print(f'列数:{df.shape[1]}')

    # dataframe合并
    # temp = pd.merge(df, df, how='left', left_on='name', right_index=True)
    # temp: pd.DataFrame = temp.drop_duplicates(['age'])

    # 保存excel
    column_index = ['name', 'age', 'sex', 'count']
    df.to_excel('../out/pandas.xlsx', columns=column_index)
    print('finished')
