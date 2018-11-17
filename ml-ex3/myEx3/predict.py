import numpy as np
from sigmoid import sigmoid


def predict(theta_1, theta_2, X):
    m, n = X.shape
    X = np.vstack((np.ones(m), X.T)).T  # 5000x401
    a2 = sigmoid(np.dot(X, theta_1.T))  # 5000x25
    m, n = a2.shape
    a2 = np.vstack((np.ones(m), a2.T)).T  # 5000x26
    prediction = sigmoid(np.dot(a2, theta_2.T))  # 5000x10
    return np.argmax(prediction, axis=1)
