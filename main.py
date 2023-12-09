from mlp_numpy.model import model as fc_model
from cnn_tf.model import model as cnn_tf_model
from resnet50_tf.model import model as resnet_model
from pretrained_resnet_tf.model import model as resnet_pretrained_model
import download_data as dd
import numpy as np
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def get_data():
    x_train, y_train, x_test, y_test = dd.load_data(flatten = True)
    num_classes = 10

    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test.flatten()] = 1

    return x_train, y_train_one_hot, x_test, y_test_one_hot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        'MNIST Destroyer is an application that trains several Deep Learning models ' + 
        'in an attempt to find the best-performing one on the MNIST dataset.' 
    )

    parser.add_argument('model_name', choices=['mlp', 'conv', 'resnet50', 'resnet50_finetuned'], help='Model to be trained')
    parser.add_argument('--learning_rate', default=0.001)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--epochs', default=10)

    args = parser.parse_args()
    model_name = args.model_name
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    epochs = args.epochs

    x_train, y_train, x_test, y_test = get_data()

    print('Using model: ', model_name)
    if model_name == 'mlp':
        fc_model(x_train, y_train.T, x_test, y_test.T, learning_rate, batch_size, epochs)
    elif model_name == 'conv':
        cnn_tf_model(x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs)
    elif model_name == 'resnet50':
        resnet_model(x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs)
    elif model_name == 'resnet50_finetuned':
        resnet_pretrained_model(x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs)

    # MODEL 2: CNN implementaion in numpy. WIP!!!
    # x_train = x_train[0:900]
    # one_hot_labels = one_hot_labels[0:900].T
    # cnn_model(x_train, one_hot_labels)
