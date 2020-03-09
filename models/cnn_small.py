import tensorflow as tf


class SmallCNN(tf.keras.Model):
    def __init__(self, out_dim):
        super(SmallCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8, kernel_size=3, padding='same', strides=1)
        self.conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, padding='same', strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same', strides=1)
        self.conv4 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding='same', strides=1)

        self.l1 = tf.keras.layers.Dense(units=out_dim)
        self.l2 = tf.keras.layers.Dense(units=out_dim)

        self.activation = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D(pool_size=2, strides=2)

        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()

    def call(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv3(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.conv4(x)
        x = self.activation(x)
        x = self.max_pool(x)

        h = self.global_pool(x)

        x = self.l1(h)
        x = self.activation(x)
        x = self.l2(x)

        return h, x
