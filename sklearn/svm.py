from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


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
    """
        C：误差项的惩罚参数C
        gamma: 核相关系数。浮点数，If gamma is ‘auto’ then 1/n_features will be used instead.
    """
    model = SVC(C=1.0, kernel='rbf', gamma ='auto')
    model.fit(x_train, y_train)
    result = model.score(x_test, y_test)
    print(f'result: {result}')