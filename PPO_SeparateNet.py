import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
import numpy as np


class Actor(tf.keras.Model):
    def __init__(self, n_actions, l1=32, l2=32, chkpt_dir='Temp/actor'):
        super(Actor, self).__init__()
        self.conv1 = Conv2D(filters=16,
                            kernel_size=(8, 8),
                            strides=4,
                            activation="relu",
                            input_shape=(4, 66, 66),
                            data_format="channels_first")
        self.conv2 = Conv2D(filters=32,
                            kernel_size=(4, 4),
                            strides=2,
                            activation="relu",
                            data_format="channels_first")
        # self.mp1 = MaxPool2D(pool_size=(2, 2),
        #                     strides=1,
        #                     data_format="channels_first")
        self.conv3 = Conv2D(filters=32,
                            kernel_size=(4, 4),
                            strides=2,
                            activation="relu",
                            data_format="channels_first")
        self.conv4 = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            activation="relu",
                            data_format="channels_first")
        self.conv5 = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            activation="relu",
                            data_format="channels_first")
        # self.mp3 = MaxPool2D(pool_size=(2, 2),
        #                     strides=1,
        #                     data_format="channels_first")
        self.flatten = Flatten()
        self.n_action = n_actions
        self.l1 = l1
        self.l2 = l2
        self.chkpt_dir = chkpt_dir

        self.layer1 = Dense(l1, activation="relu")
        self.layer2 = Dense(l2, activation="relu")
        self.output_layer = Dense(n_actions, activation=None)

    def call(self, state, training=None, mask=None):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        # print(f"State shape = {state.shape}")
        # state = tf.expand_dims(state, axis=0)
        # assert state.shape == (1, 4, 83, 96) or state.shape == (4000, 4, 83, 96),\
        # print(f"State shape before convolution {state.shape}")
        x = self.conv1(state)
        # print(f"After first convolution {x.shape}")
        x = self.conv2(x)
        # print(f"After second convolution {x.shape}")
        # x = self.mp1(x)
        # print(f"After maxpooling convolution {x.shape}")
        x = self.conv3(x)
        # print(f"After third convolution {x.shape}")
        x = self.conv4(x)
        # print(f"After forth convolution {x.shape}")
        x = self.conv5(x)
        # print(f"After fifth convolution {x.shape}")
        x = self.flatten(x)
        # print(f"After flatten layer {x.shape}")
        # exit()
        x = self.layer1(x)
        # print(f"After layer1 {x.shape}")
        x = self.layer2(x)
        # y = print(np.copy(x).max())
        return self.output_layer(x)


class Critic(tf.keras.Model):
    def __init__(self, l1=32, l2=32, chkpt_dir='Temp/critic'):
        super(Critic, self).__init__()
        self.l1 = l1
        self.l2 = l2
        self.chkpt_dir = chkpt_dir
        self.conv1 = Conv2D(filters=16,
                            kernel_size=(8, 8),
                            strides=4,
                            activation="relu",
                            input_shape=(4, 66, 66),
                            data_format="channels_first")
        self.conv2 = Conv2D(filters=32,
                            kernel_size=(4, 4),
                            strides=2,
                            activation="relu",
                            data_format="channels_first")
        self.mp1 = MaxPool2D(pool_size=(2, 2),
                             strides=1,
                             data_format="channels_first")
        self.conv3 = Conv2D(filters=32,
                            kernel_size=(4, 4),
                            strides=2,
                            activation="relu",
                            data_format="channels_first")
        self.conv4 = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            activation="relu",
                            data_format="channels_first")
        self.conv5 = Conv2D(filters=64,
                            kernel_size=(3, 3),
                            strides=1,
                            activation="relu",
                            data_format="channels_first")
        self.flatten = Flatten()
        self.layer1 = Dense(l1, activation="relu")
        self.layer2 = Dense(l2, activation="relu")
        self.output_layer = Dense(1, activation=None)

    def call(self, state, training=None, mask=None):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        # state = tf.expand_dims(state, axis=0)
        # assert state.shape == (1, 4, 83, 96) or state.shape == (4000, 4, 83, 96),\
        #     f"State shape before critic call {state.shape}"
        x = self.conv1(state)
        x = self.conv2(x)
        # x = self.mp1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        # print(f"After flatten layer {x.shape}")
        # exit()
        x = self.layer1(x)
        x = self.layer2(x)

        return self.output_layer(x)
