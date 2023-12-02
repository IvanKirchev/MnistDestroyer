from datasets import load_dataset
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def load_data():
    # Loading MNIST dataset with caching
    mnist_dataset = load_dataset('mnist', cache_dir='datasets')

    # Accessing train and test splits
    train_data = mnist_dataset['train']
    test_data = mnist_dataset['test']

    # Accessing features and labels
    train_features = np.array([np.array(i).reshape(-1) for i in train_data['image']]).T
    train_labels = np.array(train_data['label'])[:, np.newaxis]
    test_features = np.array([np.array(i).reshape(-1) for i in test_data['image']]).T
    test_labels = np.array(test_data['label'])[:, np.newaxis]

    # print(train_features.shape)  (784, 60000) 
    # print(train_labels.shape)    (60000, 1) 
    # print(test_features.shape)   (784, 10000) 
    # print(test_labels.shape)     (10000, 1)

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
