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
    # 网格搜索调参+k折交叉验证自定义实现
    from sklearn.model_selection import cross_val_score

    best_score = 0.0
    for gamma in [0.001, 0.01, 0.1, 1, 10, 100]:
        for C in [0.001, 0.01, 0.1, 1, 10, 100]:
            svm = SVC(gamma=gamma, C=C)
            # 5折交叉验证
            scores = cross_val_score(svm, X_train, y_train, cv=5)
            # 取平均数
            score = scores.mean()
            print(f'current:{score}, best:{best_score}')
            if score > best_score:
                best_score = score
                best_parameters = {"gamma": gamma, "C": C}
    svm = SVC(**best_parameters)
    svm.fit(X_train, y_train)
    test_score = svm.score(X_test,y_test)
    print("Best score on validation set:{:.2f}".format(best_score))
    print("Best parameters:{}".format(best_parameters))
    print("Score on testing set:{:.2f}".format(test_score))