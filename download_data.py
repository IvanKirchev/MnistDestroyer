from datasets import load_dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_data(flatten = True):
    # Loading MNIST dataset with caching
    mnist_dataset = load_dataset('mnist', cache_dir='datasets')

    # Accessing train and test splits
    train_data = mnist_dataset['train']
    test_data = mnist_dataset['test']

    train_features = np.array(train_data['image'])[:, :, :, np.newaxis]
    test_features = np.array(test_data['image'])[:, :, :, np.newaxis]
    
    train_labels = np.array(train_data['label'])[:, np.newaxis]
    test_labels = np.array(test_data['label'])[:, np.newaxis]

    return train_features, train_labels, test_features, test_labels
    
    # fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 4))

    # axes = axes.flatten()

    # image_idxs = np.random.random_integers(0, 50, 10)
    # print(image_idxs.shape)

    # for i in range(10):
    #     axes[i].imshow(train_features[image_idxs[i]])
    #     axes[i].set_title(train_labels[image_idxs[i]])

    # plt.tight_layout()
    # plt.show()
