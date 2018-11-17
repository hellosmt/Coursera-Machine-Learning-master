# coding=utf-8

from lrCostFunction import lrCostFunction
import numpy as np
from scipy.optimize import minimize


def oneVsAll(X, y, num_labels, _lambda):

    m, n = X.shape

    # You need to return the following variables correctly
    all_theta = np.zeros((num_labels, n+1))

    # Add ones to the X data matrix
    X = np.vstack((np.ones(m), X.T)).T
    y = y.reshape(-1, 1)

    initial_theta = np.zeros(n + 1)

    for c in range(num_labels):
        res = minimize(lrCostFunction, initial_theta, method='CG', jac=True, options={'maxiter': 50}, args=(X, y == c, _lambda))
        # 对每一种标签（0~10）循环与y进行比较，看y对应的5000个训练实例和当前标签c是否相等，即是否属于该标签类，返回True（1）或者False（0），就变成了二分类的问题
        all_theta[c] = res.x
    return all_theta
