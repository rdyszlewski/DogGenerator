from keras import Sequential
from keras.initializers import RandomNormal
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, ReLU, LeakyReLU, Dropout, Flatten, BatchNormalization
from keras.optimizers import Adam

# TODO: wstawić jakiś plik z ustawieniami
from trainers.config import TrainerConfig


class Generator:

    @staticmethod
    def create_model(input_shape, depth, noise_size):

        input_dim = noise_size
        init = RandomNormal(mean=0.0, stddev=0.02)

        model = Sequential()
        Generator.__add_input_layer(depth * 8, input_shape, input_dim, init, model)

        Generator.__add_conv_layer(depth * 8, init, model)
        model.add(Dropout(0.5))
        Generator.__add_conv_layer(depth * 4, init,  model)
        model.add(Dropout(0.5))
        Generator.__add_conv_layer(depth * 2, init, model)
        Generator.__add_conv_layer(depth, init, model)

        Generator.__add_output_layer(init, model)
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        model.summary()

        return model

    @staticmethod
    def __add_output_layer(init, model):
        model.add(Conv2D(3, kernel_size=3, activation='tanh', padding='same', kernel_initializer=init))

    @staticmethod
    def __add_input_layer(depth, input_shape, input_dim, init, model):
        factor = int(input_shape[0] / 16)
        start_shape = depth * factor * factor
        model.add(Dense(start_shape, kernel_initializer=init, input_dim=input_dim))
        model.add(Reshape((factor, factor, depth)))

    @staticmethod
    def __add_conv_layer(size, initializer, model):
        model.add(UpSampling2D())
        model.add(Conv2D(size, kernel_size=3, padding="same", kernel_initializer=initializer))
        model.add(BatchNormalization())
        model.add(ReLU())


class Discriminator:

    @staticmethod
    def create_model(input_shape, depth):
        init = RandomNormal(mean=0.0, stddev=0.02)


        model = Sequential()

        Discriminator.__add_input_layer(depth, input_shape, init, model)

        Discriminator.__add_conv_layer(depth * 2, init, model)
        Discriminator.__add_conv_layer(depth * 4, init, model)
        Discriminator.__add_conv_layer(depth * 8, init, model)
        Discriminator.__add_conv_layer(depth * 16, init, model)
        # Discriminator.__add_conv_layer(depth * 32, init, model)

        Discriminator.__add_output_layer(init, model)

        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002, beta_1=0.5))
        model.summary()

        return model

    @staticmethod
    def __add_input_layer(depth, input_shape, init, model):
        model.add(
            Conv2D(depth, kernel_size=5, strides=2, padding='same', kernel_initializer=init, input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        # model.add(Dropout(0.25))

    @staticmethod
    def __add_output_layer(init, model):
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

    @staticmethod
    def __add_conv_layer(depth, initizalizer, model):
        model.add(Conv2D(depth, kernel_size=5, strides=2, padding='same', kernel_initializer=initizalizer))
        model.add(BatchNormalization())
        model.add(LeakyReLU(0.2))
        # model.add(Dropout(0.25))
