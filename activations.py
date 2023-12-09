import numpy as np

def sigmoid(z):
  '''
    Sigmoid function

    Params:
    z: shape (n_l, m)
    '''
  return 1 / (1 + np.exp(-z))

def relu(z):
    '''
    ReLU function

    Params:
    z: shape (n_l, m)
    '''
    return np.maximum(z, 0)

def softmax(z):
    '''
    Softmax function

    Params:
    z: shape (n_l, m)
    '''
    exp_x = np.exp(z)
    return exp_x / np.sum(exp_x, keepdims=True, axis = 0)