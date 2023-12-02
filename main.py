from fc_nn_numpy.model import model, forward_prop
import download_data as dd
import numpy as np

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = dd.load_data()
    # num_classes = 10
    # one_hot_labels = np.zeros((y_train.shape[0], num_classes))
    # one_hot_labels[np.arange(y_train.shape[0]), y_train.flatten()] = 1

    # parameters = model(x_train, one_hot_labels.T, x_test, y_test)

    # np.savez('parameters.npz', **parameters)

    with np.load('parameters.npz') as parameters:
        # Computing train accuracy
        input = x_train / 255
        AL_train, cache = forward_prop(parameters, input, [128, 64, 32, 10], keep_prob=1)

        y_hat = np.argmax(AL_train, axis = 0, keepdims = True)
        y_train = y_train.T
        train_diff = np.sum(y_hat == y_train)
        m = input.shape[1]

        train_accuracy = train_diff / m
        print(f'Train Accuracy: {train_accuracy}')

        
        #Computing test accuracy
        test_input = x_test / 255
        y_test = y_test.T
        AL_test, cache = forward_prop(parameters, test_input, [128, 64, 32, 10], keep_prob=1)

        ytest_hat = np.argmax(AL_test, axis = 0, keepdims = True)
        test_diff = np.sum(ytest_hat == y_test)
        m_test = test_input.shape[1]

        test_accuracy = test_diff / m_test
        print(f'Test Accuracy: {test_accuracy}')