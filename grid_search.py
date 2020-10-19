from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


# 构建样本
def get_data():
    # X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
    X, y = make_classification(n_samples=10000, n_features=5, n_redundant=0,
                               n_clusters_per_class=1, n_classes=2, flip_y=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    return X_train, X_test, y_train, y_test


if __name__=="__main__":
    X_train, X_test, y_train, y_test = get_data()
    # 使用gridsearch实现网格搜索和k折交叉验证
    from sklearn.model_selection import GridSearchCV

    # 把要调整的参数以及其候选值 列出来；
    param_grid = {"gamma": [0.001, 0.01, 0.1, 1, 10, 100],
                  "C": [0.001, 0.01, 0.1, 1, 10, 100]}
    print("Parameters:{}".format(param_grid))

    # 实例化一个GridSearchCV类
    grid_search = GridSearchCV(SVC(), param_grid, cv=5)
    # 训练，找到最优的参数，同时使用最优的参数实例化一个新的SVC estimator。
    grid_search.fit(X_train, y_train)
    print("Test set score:{:.2f}".format(grid_search.score(X_test, y_test)))
    print("Best parameters:{}".format(grid_search.best_params_))
    print("Best score on train set:{:.2f}".format(grid_search.best_score_))
