from fc_nn_numpy.model import model as fc_model, forward_prop
from cnn_numpy.model import model as cnn_np_model
from cnn_tf.model import model as cnn_tf_model
import download_data as dd
import numpy as np

if __name__ == "__main__":

    x_train, y_train, x_test, y_test = dd.load_data(flatten = True)
    num_classes = 10

    y_train_one_hot = np.zeros((y_train.shape[0], num_classes))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    y_test_one_hot = np.zeros((y_test.shape[0], num_classes))
    y_test_one_hot[np.arange(y_test.shape[0]), y_test.flatten()] = 1

    # Uncomment the proceeding line to run the fully connected NN implemented in numpy
    # fc_model(x_train, y_train_one_hot.T, x_test, y_test_one_hot.T, training = True)

    # CNN implementaion in numpy. WIP!!!
    # x_train = x_train[0:900]
    # one_hot_labels = one_hot_labels[0:900].T
    # cnn_model(x_train, one_hot_labels)

    # CNN implemented with Tensorflow's sequential API
    cnn_tf_model(x_train, y_train_one_hot, x_test, y_test_one_hot, epochs = 1)
    