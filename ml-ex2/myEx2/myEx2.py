# coding:utf-8

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def sigmoid(z):
    return 1/(1+np.exp(-z))


def plotData(X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.ion()
    plt.figure()
    plt.plot(X[pos][:, 0], X[pos][:, 1], 'k+')
    plt.plot(X[neg][:, 0], X[neg][:, 1], 'yo')
    plt.xlabel('Exam1 score')
    plt.ylabel('Exam2 score')
    plt.legend(['y=1', 'y=0'])
    plt.show()


def costFunction(theta, X, y):
    m = len(y)
    theta = theta.reshape(-1, 1)
    s = sigmoid(np.dot(X, theta))
    first = np.dot(y.T, np.log(s))
    second = np.dot((1-y).T, np.log(1-s))
    J = -(first + second)/m
    grad = np.dot(X.T, (s - y))/m
    return J, grad.reshape(1, -1)[0]


# load data and plot data
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
plotData(X, y)

m = len(y)
y = y.reshape(-1, 1)
theta = np.zeros(3)
X = np.vstack((np.ones(m), X.T)).T
J, grad = costFunction(theta, X, y)
print "Initial J = %f" % J
print "Initial grad = "
print grad

res = minimize(costFunction, theta, method='BFGS', jac=True, options={'maxiter': 400}, args=(X, y))  # 这里的costFunction函数里的返回值包括代价值和更新的theta值，返回的theta值一定要是一维的行向量才能继续被优化函数所接收
print "The cost at theta found by fminunc:%f" % res.fun
print "The theta found by fminunc:"
print res.x