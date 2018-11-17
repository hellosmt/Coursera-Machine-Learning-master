# coding=utf-8
import numpy as np
from sigmoid import sigmoid


def lrCostFunction(theta, X, y, _lambda):
    theta = theta.reshape(-1, 1)  # 变成列向量
    _theta = np.copy(theta)  # !!! important, don't use reference
    _theta[0, 0] = 0
    m = len(y)  # number of training sets
    s = sigmoid(np.dot(X, theta))
    J = -(np.dot(y.T ,np.log(s)) + np.dot((1-y).T, np.log(1-s)))/m +\
        _lambda*np.dot(_theta.T, _theta)/(2*m)
    grad = np.dot(X.T, (s - y)) / m + _lambda * _theta / m
    return J[0], grad.reshape(1, -1)[0]
