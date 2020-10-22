from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split


def get_data():
    # 导入数据集
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target
    # # 基于mean和std的标准化
    # scaler = preprocessing.StandardScaler().fit(data)
    # scaler.transform(data)
    #
    # # 将每个特征值归一化到一个固定范围
    # scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(data)
    # scaler.transform(data)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
    from sklearn.linear_model import LogisticRegression

    # 定义逻辑回归模型
    """
        penalty：使用指定正则化项（默认：l2）
        dual: n_samples > n_features取False（默认）
        C：正则化强度的反，值越小正则化强度越大
        n_jobs: 指定线程数
        random_state：随机数生成器
        fit_intercept: 是否需要常量
    """
    model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0,
                               fit_intercept=True, intercept_scaling=1, class_weight=None,
                               random_state=None, solver='liblinear', max_iter=100, multi_class='ovr',
                               verbose=0, warm_start=False, n_jobs=1)
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f'result: {result}')

    import pickle

    # 保存模型
    with open('../out/model.pickle', 'wb') as f:
        pickle.dump(model, f)
        print('save finish')

    # 读取模型
    with open('../out/model.pickle', 'rb') as f:
        model = pickle.load(f)
        print('load success')
    model.predict(x_test)
