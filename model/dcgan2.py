from keras import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, ReLU, LeakyReLU, Dropout, Flatten
from keras.optimizers import Adam

from trainers.config import TrainerConfig


class Generator:

    @staticmethod
    def create_model(input_shape, depth):

        input_dim = TrainerConfig.noise_size

        init = RandomNormal(mean=0.0, stddev=0.02)

        # Model
        model = Sequential()

        # Start at 4 * 4
        start_shape = depth * 8 * 4 * 4
        model.add(Dense(start_shape, kernel_initializer=init, input_dim=input_dim))
        model.add(Reshape((4, 4, depth * 8)))
        # model.add(Reshape((8,8, depth * 8)))

        # Upsample => 8 * 8
        model.add(UpSampling2D())
        model.add(Conv2D(depth * 8, kernel_size=3, padding="same", kernel_initializer=init))
        model.add(ReLU())

        # Upsample => 16 * 16
        model.add(UpSampling2D())
        model.add(Conv2D(depth * 4, kernel_size=3, padding="same", kernel_initializer=init))
        model.add(ReLU())

        # Upsample => 32 * 32
        model.add(UpSampling2D())
        model.add(Conv2D(depth * 2, kernel_size=3, padding="same", kernel_initializer=init))
        model.add(ReLU())

        # Upsample => 64 * 64
        model.add(UpSampling2D())
        model.add(Conv2D(depth, kernel_size=3, padding="same", kernel_initializer=init))
        model.add(ReLU())

        # output
        model.add(Conv2D(3, kernel_size=3, activation='tanh', padding='same', kernel_initializer=init))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        model.summary()

        return model

class Discriminator:

    @staticmethod
    def create_model(input_shape, depth):

        # Random Normal Weight Initialization
        init = RandomNormal(mean=0.0, stddev=0.02)

        # Define Model
        model = Sequential()

        # Downsample ==> 32 * 32
        model.add(
            Conv2D(depth, kernel_size=3, strides=2, padding='same', kernel_initializer=init, input_shape=input_shape))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        # Downsample ==> 16 * 16
        model.add(Conv2D(depth*2, kernel_size=3, strides=2, padding='same', kernel_initializer=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        # Downsample => 8 * 8
        model.add(Conv2D(depth * 4, kernel_size=3, strides=2, padding='same', kernel_initializer=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        # Downsample => 4 * 4
        model.add(Conv2D(depth * 8, kernel_size=3, strides=2, padding='same', kernel_initializer=init))
        model.add(LeakyReLU(0.2))
        model.add(Dropout(0.25))

        # Final Layers
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))

        model.summary()

        return model