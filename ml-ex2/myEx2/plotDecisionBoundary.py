# coding=utf-8
import matplotlib.pyplot as plt
import numpy as np


#PLOTDECISIONBOUNDARY Plots the data points X and y into a new figure with
#the decision boundary defined by theta
#   PLOTDECISIONBOUNDARY(theta, X,y) plots the data points with + for the 
#   positive examples and o for the negative examples. X is assumed to be 
#   a either 
#   1) Mx3 matrix, where the first column is an all-ones column for the 
#      intercept.
#   2) MxN, N>3 matrix, where the first column is all-ones
# 网上复制的
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


def mapFeature(X1, X2):
    degree = 6
    feature = np.ones(X1.shape)
    for i in range(1, degree+1):
        for j in range(degree):
            feature = np.row_stack((feature, np.power(X1, i-j) * np.power(X2, j)))
    return feature


def plotDecisionBoundary(theta, X, y):
    
    # Create New Figure
    plotData(X[:,1:3],y.T[0])

    if X.shape[1] <= 3:
        # Only need 2 points to define a line, so choose two endpoints
        plot_x = np.array([np.min(X[:,1]), np.max(X[:,1])])

        # Calculate the decision boundary line
        plot_y = (-1/theta[2])*(theta[1]*plot_x + theta[0])

        # Plot, and adjust axes for better viewing
        plt.plot(plot_x,plot_y)

        # Legend, specific for the exercise
        plt.legend(['Admitted', 'Not admitted', 'Decision Boundary'])
        plt.axis([30, 100, 30, 100])
    else:
        # Here is the grid range
        u = np.linspace(-1, 1.5, 50)
        v = np.linspace(-1, 1.5, 50)

        z = np.zeros((u.shape[0], v.shape[0]))
        # Evaluate z = theta*x over the grid
        for i in range(0,u.shape[0]):
            for j in range(0,v.shape[0]):
                z[i,j] = np.dot(theta.T, mapFeature(u[i],v[j]))

        # !!! important for plot
        u, v = np.meshgrid(u, v)

        # Plot z = 0
        # Notice you need to specify the range [0, 0]
        plt.contour(u, v, z.T, (0,), colors='g', linewidths=2)
        plt.savefig('plotDecisionBoundaryReg.png')

    
