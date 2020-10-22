from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, validation_curve


def get_data():
    # 导入数据集
    iris = datasets.load_iris()
    data = iris.data
    target = iris.target

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, shuffle=True)
    return x_train, x_test, y_train, y_test


#  校验曲线，方便的改变模型参数，获取模型表现
if __name__ == "__main__":
    x_train, x_test, y_train, y_test = get_data()
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

    """参数
    ---
        model:用于fit和predict的对象
        X, y: 训练集的特征和标签
        param_name：将被改变的参数的名字
        param_range： 参数的改变范围
        cv：k-fold

    返回值
    ---
       train_score: 训练集得分（array）
        test_score: 验证集得分（array）
    """
    train_score, test_score = validation_curve(model, x_train, y_train, 'tol', [0.000001, 10], cv=None, scoring=None, n_jobs=1)
    print(f'train score: {train_score}')
    print(f'test score: {test_score}')

