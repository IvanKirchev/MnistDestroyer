# mnist_destroyer
My attempts at reaching 100% test accuracy on the MNIST dataset

## MLP implementaion with numpy
In this module you will find a numpy implementation of a classic MLP
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

## CNN implementation with numpy
    Work in Progress

## CNN implementation with tensorflow
input -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Dense -> SoftMax


## ResNet50 implemented with tensorflow
Con2D -> BatchNorm -> ReLU -> MaxPool -> ConvBlock -> 2xIdentityBlock -> ConvBlock -> 3xIdentity -> ConvBlock -> 5xIdentity -> ConvBlock -> 2xIdentity -> AvgPool -> Flatten -> Softmax


## ResNet50 pretrained on imageNet dataset
Replaced the last layer with a 10 units softmax layer