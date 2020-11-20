import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import make_scorer
from sklearn.utils import check_X_y


def dcg_score(x_true, y_score, k):
    pass


def ndcg_score(y_true, y_score, k):
    """
    计算ndcg得分
    :param y_true:
    :param y_score:
    :param k:
    :return:
    """

    y_score, y_true = check_X_y(y_score, y_true)



if __name__=="__main__":
    y_true = [0, 1, 0]
    y_score = [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
