import numpy as np
import math
import download_data as dd
import copy
from activations import relu, softmax
from cost_functions import categorical_cross_entropy_cost

def init_parameters(layers, input_shape):
    '''
    Parameter initialization. ReLU for the hidden layers, Softmax for the last layer

    Params:
    layers: array of the network dimentions
    input_shape: the shape of the input as a tuple
    '''
    parameters = {}
    L = len(layers)

    # He initialization for Relu layers
    parameters['W' + str(1)] = np.random.normal(0, math.sqrt(2 / layers[0]), (layers[0], input_shape))
    parameters['b' + str(1)] = np.zeros((layers[0], 1))
    
    for l in range(1, L - 1):
        parameters['W' + str(l + 1)] = np.random.normal(0, math.sqrt(2 / layers[l]), (layers[l], layers[l - 1]))
        parameters['b' + str(l + 1)] = np.zeros((layers[l], 1))

    # Xavier initialization for softmax layer
    parameters['W' + str(L)] = np.random.normal(0, 1 / (layers[L - 2] + 10), (layers[L - 1], layers[L - 2]))
    parameters['b' + str(L)] = np.zeros((layers[L - 1], 1))
    
    return parameters

def forward_prop(params, X, layers, keep_prob):
    '''
    Forward pass through the network

    params: Weights and biases for each layer
    X: Input. Shape (n, m)
    '''
    caches = []
    AL = 0
    L = len(layers)
    A_prev = X

    for l in range(1, L):
        W = params['W' + str(l)]
        b = params['b' + str(l)]
        Z = np.dot(W, A_prev) + b
        A = relu(Z)
        D = np.random.rand(A.shape[0], A.shape[1])
        D = (D < keep_prob).astype(int)
        A = A * D
        A /= keep_prob
        caches.append(((A_prev, W, b, D), Z))
        A_prev = A

    # Softmax activation at the last layer
    W = params['W' + str(L)]
    b = params['b' + str(L)]
    
    Z = np.dot(W, A_prev) + b
    A = softmax(Z)

    caches.append(((A_prev, W, b), Z))

    AL = A
    return AL, caches

def linear_activation_backward(dA, cache, m, keep_prob):
    '''
    Compute gradients for a single layer

    Params:
    dA: Gradient of activation at the current layer
    cache: tuple ((A_prev, W, b), Z)
    m: number of training examples
    '''
    ((A_prev, W, b, D), Z) = cache

    Z[Z <= 0] = 0
    Z[Z > 0] = 1

    dA = dA * D
    dA /= keep_prob

    dZ = np.multiply(dA,  Z)

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, keepdims = True, axis = 1)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def backward_prop(AL, y, caches, lambd, keep_prob):
    '''
    Compute gradients of the cost function with resepect to each parameter (W,b)

    Params:
    AL: Last layer's activations. Shape: (10, m)
    y: True labels. Shape (10, m)
    caches: Array of caches for each layer storing ((A_prev, W, b), Z)
    '''
    grads = {}
    L = len(caches)
    m = y.shape[1]


    ((A_prev, W, b), Z) = caches[-1]
    dZL = AL - y

    grads['dW' + str(L)] = (1 / m) * np.dot(dZL, A_prev.T) # + (lambd / m) * W
    grads['db' + str(L)] = (1 / m) * np.sum(dZL, keepdims = True, axis = 1)
    dA_prev = np.dot(W.T, dZL)
    
    for l in reversed(range(L - 1)):
        dA_prev, dW, db = linear_activation_backward(dA_prev, caches[l], m, keep_prob)
        grads['dW' + str(l + 1)] = dW # + (lambd / m) * W
        grads['db' + str(l + 1)] = db

    return grads

def update_parameters(grads, params, learning_rate):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(1, L + 1):
        parameters['W' + str(l)] -= learning_rate * grads['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * grads['db' + str(l)]

    return parameters

def update_parameters_momentum(grads, params, v, learning_rate, beta):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v['dW' + str(l)] = beta * v['dW' + str(l)] + (1 - beta) * grads['dW' + str(l)]
        v['db' + str(l)] = beta * v['db' + str(l)] + (1 - beta) * grads['db' + str(l)]
        parameters['W' + str(l)] -= learning_rate * v['dW' + str(l)]
        parameters['b' + str(l)] -= learning_rate * v['db' + str(l)]

    return parameters

def update_parameters_adam(grads, params, v, s, t, learning_rate, beta1, beta2, epsilon):
    parameters = copy.deepcopy(params)
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}
    
    for l in range(1, L + 1):
        v['dW' + str(l)] = beta1 * v['dW' + str(l)] + (1 - beta1) * grads['dW' + str(l)]
        v['db' + str(l)] = beta1 * v['db' + str(l)] + (1 - beta1) * grads['db' + str(l)]

        v_corrected['dW' + str(l)] = v['dW' + str(l)] / (1 - (beta1 ** t))
        v_corrected['db' + str(l)] = v['db' + str(l)] / (1 - (beta1 ** t))

        s['dW' + str(l)] = beta2 * s['dW' + str(l)] + (1 - beta2) * (grads['dW' + str(l)] ** 2)
        s['db' + str(l)] = beta2 * s['db' + str(l)] + (1 - beta2) * (grads['db' + str(l)] ** 2)

        s_corrected['dW' + str(l)] = s['dW' + str(l)] / (1 - (beta2 ** t))
        s_corrected['db' + str(l)] = s['db' + str(l)] / (1 - (beta2 ** t))

        parameters['W' + str(l)] -= learning_rate * (v_corrected['dW' + str(l)] / (np.sqrt(s_corrected['dW' + str(l)]) + epsilon))
        parameters['b' + str(l)] -= learning_rate * (v_corrected['db' + str(l)] / (np.sqrt(s_corrected['db' + str(l)]) + epsilon))
        

    return parameters, v, s, v_corrected, s_corrected

def random_mini_batches(X, Y, mini_batch_size):
    '''
    Create an array of mini batches by first randomly shuffling the 
    dataset and then partitioning it into batches

    Params:
    X: (n, m)
    Y: (10, m)
    mini_batch_size: int
    '''
    m = X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation]

    for i in range(0, m, mini_batch_size):
        end_idx = min(i + mini_batch_size, m)
        mini_batches.append((shuffled_X[:, i:end_idx], shuffled_Y[:, i:end_idx]))

    return mini_batches

def initialize_adam(parameters):
    L = len(parameters) // 2
    v = {}
    s = {}
    
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)
        s['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        s['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return v, s

def initialize_momentum(parameters):
    L = len(parameters) // 2
    v = {}
    
    for l in range(1, L + 1):
        v['dW' + str(l)] = np.zeros(parameters['W' + str(l)].shape)
        v['db' + str(l)] = np.zeros(parameters['b' + str(l)].shape)

    return v

def evaluate(x_train, y_train, x_test, y_test, parameters, layers):
    # Computing train accuracy
    AL_train, cache = forward_prop(parameters, x_train, layers, keep_prob=1)

    y_train_hat = np.argmax(AL_train, axis = 0, keepdims = True)
    y_train_truth = np.argmax(y_train, axis = 0, keepdims = True)

    train_diff = np.sum(y_train_hat == y_train_truth)
    m = x_train.shape[1]

    train_accuracy = train_diff / m
    print(f'Train Accuracy: {train_accuracy}')

    #Computing test accuracy
    AL_test, cache = forward_prop(parameters, x_test, layers, keep_prob=1)

    y_test_hat = np.argmax(AL_test, axis = 0, keepdims = True)
    y_test_truth = np.argmax(y_test, axis = 0, keepdims = True)

    test_diff = np.sum(y_test_hat == y_test_truth)
    m_test = x_test.shape[1]

    test_accuracy = test_diff / m_test
    print(f'Test Accuracy: {test_accuracy}')

def model(x_train, y_train, x_test, y_test, learning_rate = 0.001, 
          batch_size = 256, epochs = 1, layers = [128, 64, 32, 10], lambd = 0.01, 
          keep_prob = 0.9, beta = 0.9, beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    '''
    Training of an MLP model. Features of the model are: mini-batch GD with Adam, 
    dropout regularization, L2 regularization. For using Momentum replace 
    Adam init on line #276 and Adam iters on line #297-298 with the Momentum related code. 

    Params:
    x_train: tuple (m, 28, 28, 1)
    y_train: tuple (10, m)
    x_test: tuple (t, 28, 28, 1)
    y_test; tuple (10, t)
    
    learning_rate: scaler
    epochs: scaler
    batch_size: scaler
    layers: array - Dimentions of the layers
    lambd: scaler - Regularization parameter
    keep_prob: scaler - Dropout regularization parameter
    beta: scaler - Momentum optimizer parameter. N
    beta1: scaler - Adam optimizer parameter
    beta2: scaler - Adam optimizer parameter
    epsilon: sclaer - Adam optimizer parameter
    '''

    # Reshape so m (num of examples) is at the last axis
    m_train = x_train.shape[0]
    m_test = x_test.shape[0]
    x_train = x_train.reshape(m_train, -1).T
    x_test = x_test.reshape(m_test, -1).T

    params = init_parameters(layers, x_train.shape[0])
    x_train = x_train / 255
    x_test = x_test / 255

    v, s = initialize_adam(params)
    # v = initialize_momentum(params)
    t = 0

    for e in range(epochs):
        mini_batches = random_mini_batches(x_train, y_train, batch_size)
        if e > 15:
            learning_rate = 0.0001
        if e > 20:
            learning_rate = 0.00001

        for i in range(len(mini_batches)):
            x_batch = mini_batches[i][0]
            y_batch = mini_batches[i][1]

            AL, caches = forward_prop(params, x_batch, layers, keep_prob)

            grads = backward_prop(AL, y_batch, caches, lambd, keep_prob)
            
            # Adam optimizer
            t = t + 1
            params, v, s, _, _ = update_parameters_adam(grads, params, v, s, t, learning_rate, beta1, beta2,  epsilon)
            
            # Regular Mini-batch GD
            # params = update_parameters(grads, params, learning_rate)

            # Mini-batch GD with momentum
            # params = update_parameters_momentum(grads, params, v, learning_rate, beta)

            if i % 20 == 0:
                loss = categorical_cross_entropy_cost(AL, y_batch, params, lambd)
                print(f'Train loss at epoch {e} batch {i} is {loss}')
                # print(f'Max gradient at layer 1: {np.max(grads['dW1'][0])}')

    evaluate(x_train, y_train, x_test, y_test, params, layers)
