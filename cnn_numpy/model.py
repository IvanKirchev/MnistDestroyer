import numpy as np
import math
from activations import sigmoid, softmax
from cost_functions import categorical_cross_entropy_cost

def zero_pad(X, pad):
    """
    Pad with zeros all images of the dataset X. The padding is applied to the height and width of an image
    
    Argument:
    X -- python numpy array of shape (m, n_H, n_W, n_C) representing a batch of m images.
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions.
    
    Returns:
    X_pad -- padded image of shape (m, n_H + 2 * pad, n_W + 2 * pad, n_C)
    """
    return np.pad(array = X, pad_width = ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode = 'constant', constant_values = 0)

def conv_single_step(a_slice_prev, W, b):
    """
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    Z -- a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    """

    S = np.multiply(a_slice_prev, W)
    Z = np.sum(S) + float(b.squeeze())
    return Z

def conv_forward(A_prev, W, b, hparameters):
    """
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
        numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    (f, f, n_C_prev, n_C) = W.shape

    pad = hparameters['pad']
    stride = hparameters['stride']

    n_H = math.floor((n_H_prev + (2 * pad) - f) / stride) + 1
    n_W = math.floor((n_W_prev + (2 * pad) - f) / stride) + 1

    Z = np.zeros((m, n_H, n_W, n_C))

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start: vert_end, horiz_start:horiz_end, :]
                    weights = W[:, :, :, c]
                    biases = b[:, :, :, c]

                    Z[i, h, w, c] = conv_single_step(a_slice_prev, weights, biases)

    cache = (A_prev, W, b, hparameters, Z)
    # A = sigmoid(Z)
    return Z, cache

def max_pool_forward(A_prev, hparameters):
    """
    Implements the forward pass of the pooling layer
    
    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")
    
    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """
    
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f

            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f

                for c in range(n_C):
                    a_slice_prev = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                    A[i, h, w, c] = np.max(a_slice_prev)

    cache = (A_prev, hparameters)
    return A, cache

# x = np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,188,255,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,191,250,253,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,248,253,167,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,247,253,208,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,29,207,253,235,77,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,54,209,253,253,88,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,93,254,253,238,170,17,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,210,254,253,159,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,209,253,254,240,81,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,27,253,253,254,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,206,254,254,198,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,168,253,253,196,7,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,20,203,253,248,76,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,22,188,253,245,93,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,103,253,253,191,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,240,253,195,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,15,220,253,253,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,94,253,253,253,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,89,251,253,250,131,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,214,218,95,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
# x = x.reshape(1, 28, 28, 1)

# # 3 x 3 x 28 x 2
# W = np.random.rand(3, 3, 28, 2)
# b = np.ones((1, 1, 1, 2))
# A, Z, cache = conv_forward(x, W, b, {'pad': 1, 'stride': 3})

# print(x.shape)
# print(Z.shape)

def init_filters(filter_size, n_C, n_C_prev):
    weights = np.random.normal(0, 2 / (n_C + n_C_prev), (filter_size, filter_size, n_C_prev, n_C))
    biases = np.zeros((1, 1, 1, n_C))
    return weights, biases

def init_params_softmax(c, n_a_prev):
    W = np.random.normal(0, 1 / (n_a_prev + c), (c, n_a_prev))
    b = np.zeros((c, 1))

    return W, b

def forward_pass(x_train, y_train, conv_W, conv_b, W, b):
    m = x_train.shape[0]
    caches = []

    Z1, cache = conv_forward(x_train, conv_W, conv_b, {'pad': 1, 'stride': 3})
    caches.append(cache)

    A1 = sigmoid(Z1)

    A2, cache = max_pool_forward(A1, {'f': 5, 'stride': 3})
    caches.append(cache)

    A2 = A2.reshape(m, -1)

    Z3 = np.dot(W, A2.T) + b
    A3 = softmax(Z3)

    caches.append((A2.T, W, b, Z3))

    return A3, caches

def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.
    
    Arguments:
    x -- Array of shape (f, f)
    
    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """    
    max_val = np.max(x)
    mask = (x == max_val)

    return mask

def backward_prop(A3, y, caches):
    grads = {}
    L = len(caches)
    m = y.shape[1]


    ((A2, W3, b3, Z3)) = caches[-1]
    dZ3 = A3 - y

    dW3 = (1 / m) * np.dot(dZ3, A2.T)
    db3 = (1 / m) * np.sum(dZ3, axis = 1, keepdims = True)

    dA2 = np.dot(W3.T, dZ3)

    # Start maxpool backprop
    ((A1, hparameters)) = caches[-2]
    stride = hparameters['stride']
    f = hparameters['f']
    
    m, n_H1, n_W1, n_C1 = A1.shape
    m, n_H2, n_W2, n_C2 = dA2.shape

    dA1 = np.zeros(A1.shape)

    for i in range(m):
        a1 = A1[i]
        for h in range(n_H2):                   # loop on the vertical axis
            for w in range(n_W2):               # loop on the horizontal axis
                for c in range(n_C2):           # loop over the channels (depth)

                    # Find the corners of the current "slice" (≈4 lines)
                    vert_start = h
                    vert_end = h + f
                    horiz_start = w
                    horiz_end = w + f
                        
                    # Use the corners and "c" to define the current slice from a_prev (≈1 line)
                    a1_slice = a1[ vert_start: vert_end, horiz_start: horiz_end, c]
                    
                    # Create the mask from a_prev_slice (≈1 line)
                    mask = create_mask_from_window(a1_slice)

                    # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
                    dA1[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA2[i, h, w, c]

    # Start Conv backprop
    (A0, W1, b1, hparameters, Z1) = caches[-3]
    (m, n_H_prev, n_W_prev, n_C_prev) = A0.shape
    (f, f, n_C_prev, n_C) = W1.shape
    
    stride = hparameters['stride']
    pad = hparameters['pad']
    
    (m, n_H, n_W, n_C) = Z1.shape
    
    dA = None                          
    dW = None
    db = None
    
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)
    
    for i in range(m):                       # loop over the training examples
        
        a_prev_pad = None
        da_prev_pad = None
        
        for h in range(n_H):                   # loop over vertical axis of the output volume
           for w in range(n_W):               # loop over horizontal axis of the output volume
               for c in range(n_C):           # loop over the channels of the output volume
                    
                    vert_start = None
                    vert_end = None
                    horiz_start = None
                    horiz_end = None

                    a_slice = None

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += None
                    dW[:,:,:,c] += None
                    db[:,:,:,c] += None
                    
        dA_prev[i, :, :, :] = None
                    


def model(x_train, y_train, learning_rate = 0.001):
    x_train = x_train / 255
    m = x_train.shape[0]
    weights, biases = init_filters(5, 4, x_train.shape[-1])

    W3, b3 = init_params_softmax(10, 12)

    for i in range(1000):

        A3, caches = forward_pass(x_train, y_train, weights, biases, W3, b3)
        cost = categorical_cross_entropy_cost(A3, y_train)
        print("Output shape: ", A3.shape)
        print("Cost: ", cost)

        grads = backward_prop(A3, y_train, caches)

        weights, biases, W3, b3 = update_parameters(weights, biases, W3, b3, grads, learning_rate)