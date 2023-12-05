import tensorflow as tf
import wandb


def model(x_train, y_train, x_test, y_test, epochs):
    x_train = x_train / 255
    x_test = x_test / 255

    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(15, 10, padding = 'same', input_shape = x_train.shape[1:]),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(20, 8, 2, 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Conv2D(25, 6, 2, 'same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.ReLU(),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units = 30, activation = 'relu'),
        tf.keras.layers.Dense(10, activation = 'linear')
    ], 'mnist-destroyer')

    model.compile(optimizer = tf.optimizers.Adam(learning_rate = 0.005), 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits = True), 
                  metrics=['accuracy']) 
    model.fit(x_train, y_train, batch_size = 64, epochs = epochs)

    (_, accuracy) = model.evaluate(x_test, y_test)
    print("Test accuracy: ", accuracy)