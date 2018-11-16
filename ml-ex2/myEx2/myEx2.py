# coding=utf-8

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import minimize


def sigmoid(z):
    return 1/(1+np.exp(-z))


def plotData(X, y):
    plt.ion()  # 打开交互模式
    plt.figure()
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.plot(X[pos][:, 0], X[pos][:, 1], 'k+', markersize=7)
    plt.plot(X[neg][:, 0], X[neg][:, 1], 'ko', color='y')
    plt.xlabel('Exam1 score')
    plt.ylabel('Exam2 score')
    plt.legend(['Admitted', 'Not admitted'])
    plt.savefig("plotData.png")
    plt.show()
    input()


def costFunction(theta, X, y):
    m = len(y)
    theta = theta.reshape(-1, 1)
    s = sigmoid(np.dot(X, theta))
    first = np.dot(y.T, np.log(s))
    second = np.dot((1-y).T, np.log(1-s))
    J = -(first + second)/m
    grad = np.dot(X.T, (s - y))/m
    return J, grad.reshape(1, -1)[0]


def predict(theta, X):
    return np.round(sigmoid(np.dot(X, theta)))  # np.round()四舍五入


# load data and plot data
data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
plotData(X, y)

#get the initial J and grad witn theta = all zeros
m = len(y)
y = y.reshape(-1, 1)
theta = np.zeros(3)
X = np.vstack((np.ones(m), X.T)).T
J, grad = costFunction(theta, X, y)
print("Initial J = %f" % J)
print ("Initial grad = ")
print (grad)

# find cost and theta by fminunc
res = minimize(costFunction, theta, method='BFGS', jac=True, options={'maxiter': 400}, args=(X, y))  # 这里的costFunction函数里的返回值包括代价值和更新的theta值，返回的theta值一定要是一维的行向量才能继续被优化函数所接收
print ("The cost at theta found by fminunc:%f" % res.fun)
print ("The theta found by fminunc:")
print (res.x)

# prediction witn 45 and 85 score
theta = res.x.reshape(-1, 1)
prob = sigmoid(np.dot([1, 45, 85], theta))
print('For a student with scores 45 and 85, we predict an admission probability of %f\n\n' % prob[0])

# compute the accuracy with our training set
p = predict(theta, X)
accuracy = np.mean(np.double(p == y))*100  # np.mean()求M*N个数的平均值（asix不取值，若取0压缩行，取1压缩列）
print ("the accruracy is %.2f%%" % accuracy)


