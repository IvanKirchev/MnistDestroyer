# mnist_destroyer
My attempts at reaching 100% test accuracy on the MNIST dataset

## Fully connected Neaural Net implementaion with numpy (fc_nn_numpy)
In this module you will find a numpy implementation of a fully connected neural network
with the following architecture: 

input -> 128 -> 64 -> 32 -> 10
        relu   relu  relu   softmax

The implememntaion includes: 
    Input normalization
    He (for ReLU) and Xavier (for Softmax) parameter initializations
    Forward and Backward propagation with Categorical Cross-Entropy Loss function
    ReLU and Softmax (for the last layer) activations
    Optimization (choose only one):
        Mini-batch gradient descent
        Momentum
        Adam
    Learning rate decay
    Regularization:
        L2
        Dropout

Performance:
    Best performance: 
        Train acc: 99.6817%
        Test acc: 97.85%
