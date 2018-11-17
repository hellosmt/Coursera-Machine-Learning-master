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
from predict import predict

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

## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized
# neural network parameters.

print('\nLoading Saved Neural Network Parameters ...\n')
data = sio.loadmat('ex3weights.mat')
theta_1 = data['Theta1']  # 25x401
theta_2 = data['Theta2']  # 10x26
# print(theta_1.shape)
# print(theta_2.shape)

# ================= Part 3: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.
prediction = (predict(theta_1, theta_2, X)+1) % 10  # 因为十个类别参数时按照1234567890的顺序放的，
# 所以如果某个实例预测出来的那一行概率是坐标为9的那个数字最大，则这张图片被预测为是0
print(prediction)

