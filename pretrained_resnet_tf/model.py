import tensorflow as tf
import keras.applications.resnet_v2
import numpy as np

def model(x_train, y_train, x_test, y_test, learning_rate, batch_size, epochs):
    '''
    Fine-tuning Resnet50v2 model pre-trained on ImageNet dataset

    Params:
    x_train: (m, 28, 28, 1)
    y_train: (m, 10)
    x_test: (t, 28, 28, 1)
    y_test: (t, 10)

    learning_rate: scaler
    epochs: scaler
    batch_size: scaler
    '''
    base_model = tf.keras.applications.ResNet50V2(weights = 'imagenet', include_top = False, input_shape= (32, 32, 3))
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape = (32, 32, 3))

    x = base_model(inputs, training = False)
    x = tf.keras.layers.Flatten()(x)

    outputs = tf.keras.layers.Dense(units = 10, activation = 'linear')(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate), loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

    x_train = np.pad(x_train, pad_width = ((0, 0), (2, 2), (2, 2), (1,1)), mode = 'constant', constant_values = 0)
    x_train_preproc = tf.keras.applications.resnet_v2.preprocess_input(x_train, data_format = 'channels_last')

    model.fit(x_train_preproc, y_train, epochs, batch_size)
    
    (loss, accuracy) = model.evaluate(x_test, y_test)
    print(f'Test loss: {loss}. Test accuracy: {accuracy}')