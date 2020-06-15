from keras import Input, Model, Sequential
from keras.layers import Dense, Reshape, concatenate, Conv2D, Flatten, LeakyReLU, Dropout, Activation, \
    BatchNormalization, UpSampling2D, Conv2DTranspose
import numpy as np

class Discriminator:

    @staticmethod
    def create_model(input_shape, depth):
        dropout = 0.5
        alpha = 0.2

        model = Sequential()
        Discriminator.__add_conv_layer(model, depth, alpha, dropout, input_shape)
        Discriminator.__add_conv_layer(model, depth * 2, alpha, dropout)
        Discriminator.__add_conv_layer(model, depth * 4, alpha, dropout)
        Discriminator.__add_conv_layer(model, depth * 8, alpha, dropout)
        Discriminator.__add_output(model)

        return model

    @staticmethod
    def __add_conv_layer(model, units,  alpha, dropout, input_shape=None):
        if input_shape:
            model.add(Conv2D(units, 2, input_shape=input_shape, padding='same'))
        else:
            model.add(Conv2D(units * 2, 2))
        model.add(LeakyReLU(alpha))
        model.add(Dropout(dropout))

    @staticmethod
    def __add_output(model):
        model.add(Flatten())
        model.add(Dense(1))
        # model.add(Dense(116))
        model.add(Activation('sigmoid'))

class Generator:

    @staticmethod
    def create_model(input_shape, depth):
        # TODO: refaktoryzacja
        depth = depth * 8
        dropout = 0.4
        model = Sequential()
        dim = 32 # TODO; będzie to trzeba jakoś odpowiednio podać w parametrze
        momentum = 0.9
        # TODO: sprawdzić, skąd bierze się ta wartość
        model.add(Dense(dim * dim * depth, input_dim=100)) # TODO: to też powinno być w parametrze
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(Reshape((dim, dim, depth)))
        model.add(Dropout(dropout))

        # model.add(UpSampling2D(size=(4,4)))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        model.add(Conv2DTranspose(int(depth/4),5, padding='same'))
        model.add(BatchNormalization(momentum=momentum))
        model.add(Activation('relu'))
        # model.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        # model.add(BatchNormalization(momentum=momentum))
        # model.add(Activation('relu'))

        # TODO: sprawdzić, czy nie będzie trzeba zmienić tej jedynki
        model.add(Conv2DTranspose(3, 5, padding='same'))
        model.add(Activation('sigmoid'))
        # model.summary()

        return model



class Discriminator2:

    @staticmethod
    def create_model(input_shape, output_size):
        input_size = input_shape[0] * input_shape[1] * 3
        layer_shape = Input((input_size,))
        output_shape = Input((output_size,))
        model = Discriminator2.__create_discrimination_function(input_size, layer_shape, output_shape)
        model_discriminator = Model([layer_shape, output_shape], model)
        model_discriminator.get_layer('layer_1').trainable = False
        model_discriminator.get_layer('layer_1').set_weights([np.array([[[[-1.0]]],[[[1.0]]]])])
        model_discriminator.summary()
        model_discriminator.compile(optimizer='adam', loss='binary_crossentropy')

    @staticmethod
    def __create_discrimination_function(input_layer_size, input_shape, output_shape):
        # TODO: sprawdiź


        input_layer = Dense(input_layer_size, activation='sigmoid')(output_shape)
        input_layer = Reshape((2, input_layer_size, 1))(concatenate([input_shape, input_layer]))
        discriminator = Conv2D(filters=1, kernel_size=[2, 1], use_bias=False, name='layer_1')(input_layer)
        out = Flatten()(discriminator)
        return out


class Generator2:

    @staticmethod
    def create_model(noise_shape, input_size, output_size):
        input_layer = Input(noise_shape)
        generated = Dense(input_size, activation='linear')(input_layer)

        model = Model(input=input_layer, outputs=[generated, Reshape((output_size,))(input_layer)])
        model.summary()

        return model