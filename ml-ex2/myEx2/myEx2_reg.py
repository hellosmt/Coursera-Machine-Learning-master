import numpy as np
from matplotlib import pyplot as plt


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


# load data and plot data
data = np.loadtxt("ex2data2.txt", delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
plotData(X, y)


