import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# 构建样本
def get_data():
    # X为样本特征，y为样本类别输出， 共10000个样本，每个样本20个特征，输出有2个类别，没有冗余特征，每个类别一个簇
    X, y = make_classification(n_samples=10000, n_features=5, n_redundant=0,
                               n_clusters_per_class=1, n_classes=2, flip_y=0.1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    dtrain = xgb.DMatrix(X_train, y_train)
    dtest = xgb.DMatrix(X_test, y_test)
    return dtrain, dtest


if __name__=="__main__":
    dtrain, dtest = get_data()
    # xgboost分类模型
    param = {'max_depth': 5, 'eta': 0.5, 'verbosity': 1, 'objective': 'binary:logistic'}
    raw_model = xgb.train(param, dtrain, num_boost_round=5)
    pred_train_raw = raw_model.predict(dtrain)
    for i in range(len(pred_train_raw)):
        if pred_train_raw[i] > 0.5:
            pred_train_raw[i] = 1
        else:
            pred_train_raw[i] = 0
    print(f'训练集准确率:{accuracy_score(dtrain.get_label(), pred_train_raw)}')

    pred_test_raw = raw_model.predict(dtest)
    for i in range(len(pred_test_raw)):
        if pred_test_raw[i] > 0.5:
            pred_test_raw[i] = 1
        else:
            pred_test_raw[i] = 0
    print(f'测试集准确率:{accuracy_score(dtest.get_label(), pred_test_raw)}')

