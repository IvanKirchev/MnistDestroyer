from tensorflow.keras.initializers import random_uniform, glorot_uniform
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, BatchNormalization
import tensorflow as tf

def identity_block(X, f, filters, initializer = random_uniform):
    
    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = f, strides = 1, padding = 'same', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, s = 2, initializer=glorot_uniform):
    
    F1, F2, F3 = filters
    X_shortcut = X

    #Main Path
    X = Conv2D(filters = F1, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F2, kernel_size = f, strides = 1, padding = 'same', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    X = Conv2D(filters = F3, kernel_size = 1, strides = 1, padding = 'valid', kernel_initializer = initializer())(X)
    X = BatchNormalization(axis = 3)(X)

    #Shortcut Path
    X_shortcut = X = Conv2D(filters = F3, kernel_size = 1, strides = (s, s), padding = 'valid', kernel_initializer = initializer())(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    # Combine
    X = Add()([X_shortcut, X])

    X = Activation('relu')(X)

    return X


def model(x_train, y_train, x_test, y_test, learning_rate, epochs, batch_size):
    #Normilize input
    x_train = x_train / 255
    x_test = x_test / 255

    X_input = Input(x_train.shape[1:])

    X = ZeroPadding2D((39, 39))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2), kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = convolutional_block(X, f = 3, filters = [64, 64, 256], s = 1)
    X = identity_block(X, 3, [64, 64, 256])
    X = identity_block(X, 3, [64, 64, 256])

    X = convolutional_block(X, f = 3, filters = [128, 128, 512], s = 2)
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])
    X = identity_block(X, 3, [128, 128, 512])

    X = convolutional_block(X, f = 3, filters = [256, 256, 1024], s = 2)
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])
    X = identity_block(X, 3, [256, 256, 1024])

    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], s = 2)
    X = identity_block(X, 3, [512, 512, 2048])
    X = identity_block(X, 3, [512, 512, 2048])

    X = AveragePooling2D()(X)

    X = Flatten()(X)
    X = Dense(units = 10, activation = 'softmax', kernel_initializer = glorot_uniform(seed=0))(X)

    model = tf.keras.models.Model(inputs = X_input, outputs = X)
    
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x_train, y_train, batch_size, epochs)
    print(model.evaluate(x_test, y_test))