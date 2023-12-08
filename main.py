from mlp_numpy.model import model as fc_model, forward_prop
from cnn_numpy.model import model as cnn_np_model
from cnn_tf.model import model as cnn_tf_model
from resnet50_tf.model import model as resnet_model
from pretrained_resnet_tf.model import model as renet_pretrained_model
import download_data as dd
import numpy as np
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def get_data():
    x_train, y_train, x_test, y_test = dd.load_data(flatten = True)
    num_classes = 10

    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test.flatten()] = 1

    return x_train, y_train_one_hot, x_test, y_test_one_hot

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Mnist Destryer is a set of Deep Learning models trained on the " + 
        "MNIST dataset in an attempt to find the best performing one"
    )

    parser.add_argument("model_name", choices=['MLP', 'Conv', 'ResNet50', 'ResNet50_fine_tuned'])

    args = parser.parse_args()

    x_train, y_train, x_test, y_test = get_data()
    print("Using model: ", args.model_name)
    if args.model_name == 'MLP':
        fc_model(x_train, y_train.T, x_test, y_test.T, training = True)
    elif args.model_name == 'ConvNet':
        cnn_tf_model(x_train, y_train, x_test, y_test, epochs = 1)
    elif args.model_name == 'ResNet50':
        resnet_model(x_train, y_train, x_test, y_test, 0.001, 10, 256)
    elif args.model_name == 'ResNet50_fine_tuned':
        renet_pretrained_model(x_train, y_train, x_test, y_test)

    # MODEL 2: CNN implementaion in numpy. WIP!!!
    # x_train = x_train[0:900]
    # one_hot_labels = one_hot_labels[0:900].T
    # cnn_model(x_train, one_hot_labels)
