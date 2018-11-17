# coding=utf-8

## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#


import numpy as np
import scipy.io as sio
from displayData import displayData
from matplotlib import pyplot as plt
from lrCostFunction import lrCostFunction
from oneVsAll import oneVsAll
from predictOneVsAll import predictOneVsAll

plt.ion()
# Initialization

# Setup the parameters you will use for this part of the exercise
input_layer_size = 400   # 20x20 Input Images of Digits
num_labels = 10           # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

# =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...\n')

data = sio.loadmat('ex3data1.mat')  # training data stored in arrays X, y
X = data['X']  # .mat文件里的数据时按照矩阵形式存的，且已命名好，直接用矩阵名取就好
y = data['y']%10
# np.set_printoptions(threshold=5000) #当array中的元素个数小于threshold时，元素会全部打印出来，而不是用省略号代替
# print(y==1)
# print(1-(y==1))
m = X.shape[0]  # 5000
rand_indices = np.random.permutation(m)  # permutation对5000个数字洗牌打乱顺序，然后返回打乱后的新数组，不改变原数组

# Randomly select 100 data points to display
sel = X[rand_indices[0:100]]

displayData(sel)

input('Program paused. Press enter to continue.\n')

# ============ Part 2a: Vectorize Logistic Regression ============  #未实现
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

# Test case for lrCostFunction
# print('\nTesting lrCostFunction() with regularization')
#
# theta_t = [-2, -1, 1, 2]
# X_t = [np.ones(5,1) reshape(1:15,5,3)/10]
# y_t = [1, 0, 1, 0, 1]
# lambda_t = 3
# J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
#
# print('\nCost: \n', J)
# print('Expected cost: 2.534819\n')
# print('Gradients:\n')
# print(' #f \n', grad)
# print('Expected gradients:\n')
# print(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n')
#
# input('Program paused. Press enter to continue.\n')
# ## ============ Part 2b: One-vs-All Training ============
print('\nTraining One-vs-All Logistic Regression...\n')

_lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, _lambda)


input('Program paused. Press enter to continue.\n')


## ================ Part 3: Predict for One-Vs-All ================

pred = predictOneVsAll(all_theta, X)

print('\nTraining Set Accuracy: \n', np.mean(np.double(pred == y.T)) * 100)


