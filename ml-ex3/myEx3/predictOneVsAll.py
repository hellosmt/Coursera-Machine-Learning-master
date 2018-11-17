import numpy as np
from sigmoid import sigmoid

def predictOneVsAll(theta, X):
    m, n = X.shape
    X = np.vstack((np.ones(m), X.T)).T
    prediction = sigmoid(np.dot(theta, X.T))
    return np.argmax(prediction, axis=0)
