from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def get_data():
    # 导入数据集
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


#  k折交叉验证
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
    """
        C：误差项的惩罚参数C
        gamma: 核相关系数。浮点数，If gamma is ‘auto’ then 1/n_features will be used instead.
    """
    model = SVC(C=1.0, kernel='rbf', gamma ='auto')
    from sklearn.model_selection import cross_val_score

    """
        model：拟合数据的模型
        cv ： k-fold
        scoring: 打分参数-‘accuracy’、‘f1’、‘precision’、‘recall’ 、‘roc_auc’、'neg_log_loss'等等
    """
    result = cross_val_score(model, x_train, y_train, scoring=None, cv=5, n_jobs=1)
    print(f'result: {result}')