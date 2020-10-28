from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split


def get_data():
    # 导入数据集
    bostob = datasets.load_boston()
    data = bostob.data
    target = bostob.target
    # # 基于mean和std的标准化
    # scaler = preprocessing.StandardScaler().fit(data)
    # scaler.transform(data)
    #
    # # 将每个特征值归一化到一个固定范围
    # X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    # X_scaled = X_std * (max - min) + min
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
    # scaler.transform(data)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


if __name__=="__main__":
    x_train, x_test, y_train, y_test = get_data()

    from sklearn.linear_model import LinearRegression

    # 定义线性回归模型
    """
        fit_intercept：是否计算截距。False-模型没有截距
        normalize： 当fit_intercept设置为False时，该参数将被忽略。 如果为真，则回归前的回归系数X将通过减去平均值并除以l2-范数而归一化。
        n_jobs：指定线程数
    """
    model = LinearRegression(fit_intercept=True, normalize=False,
                             copy_X=True, n_jobs=1)

    # 拟合模型
    model.fit(x_train, y_train)
    # 模型预测
    # model.predict(x_test)

    # 获得这个模型的参数
    param = model.get_params()
    print(f'param: {param}')
    # 为模型进行打分
    result = model.score(x_test, y_test)  # 线性回归：R square； 分类问题： acc
    print(f'result: {result}')
