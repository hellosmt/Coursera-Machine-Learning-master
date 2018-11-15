# coding=utf-8
import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from plotDecisionBoundary import  plotDecisionBoundary

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
    plt.savefig("plotData_reg.png")
    plt.show()


def mapFeature(X1, X2):
    degree = 6
    feature = np.ones(X1.shape)
    for i in range(1, degree+1):
        for j in range(degree):
            feature = np.row_stack((feature, np.power(X1, i-j) * np.power(X2, j)))
    return feature


def costFunctionReg(theta, X, y, _lambda):
    m = len(y)
    theta = theta.reshape(-1, 1)
    _theta = np.copy(theta)
    _theta[0, 0] = 0  # 在计算正则化损失时第一个theta为0
    s = sigmoid(np.dot(X, theta))
    first = np.dot(y.T, np.log(s))
    second = np.dot((1 - y).T, np.log(1 - s))
    third = np.dot(_theta.T, _theta)
    J = -(first + second) / m + third * _lambda / m/2
    grad = np.dot(X.T, (s - y)) / m + (_lambda * _theta) / m
    return J, grad.reshape(1, -1)[0]


# load data and plot data
data = np.loadtxt("ex2data2.txt", delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
plotData(X, y)

X = mapFeature(X[:, 0], X[:, 1]).T
y = y.reshape(-1, 1)
theta = np.zeros(X.shape[1])

_lambda = 1
J, grad = costFunctionReg(theta, X, y, _lambda)

print "Initial J by regularizatuon = %f" % J
print "Initial grad = "
print grad

# find J by fminunc
res = minimize(costFunctionReg, theta, method='BFGS', jac=True, options={'maxiter': 400}, args=(X, y, _lambda))
print "The cost at theta found by fminunc:%f" % res.fun
print "The theta found by fminunc:"
print res.x

theta = res.x.reshape(-1, 1)

plotDecisionBoundary(theta, X, y)