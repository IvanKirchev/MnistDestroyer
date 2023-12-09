# MNIST Destroyer
My attempts at reaching 100% test accuracy on the MNIST dataset

## Usage
To start traning any of the models run the following command
```bash
python3 main.py [mlp|conv|resnet50|resnet50_finetuned] 
```

The following parameters are avaibale for tuning throught the CMD using the options: learning_rate, batch_size and epochs
```bash
python3 main.py [mlp|conv|resnet50|resnet50_finetuned] --learning_rate=0.001 --batch_size=128 --epochs=10 
```

## MLP implementation with numpy (mlp)
In this module, you will find a numpy implementation of a classic MLP with the following architecture:

Input -> 128 -> 64 -> 32 -> 10
       relu   relu  relu   softmax

### Model implementation:
- Input normalization
- He (for ReLU) and Xavier (for Softmax) parameter initializations
- Forward and Backward propagation with Categorical Cross-Entropy Loss function
- ReLU and Softmax (for the last layer) activations
- Optimization (choose only one):
  - Mini-batch gradient descent
  - Momentum
  - Adam
- Learning rate decay
- Regularization:
  - L2
  - Dropout

## CNN implementation with numpy
Work in Progress

## CNN implementation with tensorflow (conv)
ConvNet with the following architecture:

Input -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Dense -> SoftMax

## ResNet50 implemented with tensorflow (resnet50)
Tensorflow implementation of ResNet50:

Input -> Conv2D -> BatchNorm -> ReLU -> MaxPool -> ConvBlock -> 2xIdentityBlock -> ConvBlock -> 3xIdentityBlock -> ConvBlock -> 5xIdentityBlock -> ConvBlock -> 2xIdentityBlock -> AvgPool -> Flatten -> Softmax

## ResNet50 pretrained on ImageNet dataset (resnet50_finetuned)
Fine-tuning a ResNet50 model pre-trained on the ImageNet dataset
