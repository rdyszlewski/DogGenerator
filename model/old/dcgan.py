from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, GlobalAveragePooling2D, Dense, Deconvolution2D, Reshape

from trainers.config import TrainerConfig


class Discriminator:

    @staticmethod
    def create_model(input_shape, depth):
        # input is an image with shape spatial_dim x spatial_dim and 3 channels
        filter_size = 5
        inp = Input(shape=input_shape)

        # design the discrimitor to downsample the image 4x
        x = Discriminator.__add_block(inp, depth, filter_size)
        x = Discriminator.__add_block(x, depth * 2, filter_size)
        x = Discriminator.__add_block(x, depth * 4, filter_size)
        x = Discriminator.__add_block(x, depth * 8, filter_size)

        # average and return a binary output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1, activation='sigmoid')(x)

        return Model(inputs=inp, outputs=x)

    @staticmethod
    def __add_block(x, filters, filter_size):
        x = Conv2D(filters, filter_size, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(filters, filter_size, padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        return x

class Generator:

    @staticmethod
    def create_model(input_shape, depth):
        filter_size = 5
        latent_dim = TrainerConfig.noise_size # TODO: wstawić to do stełej
        inp = Input(shape=(latent_dim,))

        # projection of the noise vector into a tensor with
        # same shape as last conv layer in discriminator
        x = Dense(4 * 4 * (depth * 8), input_dim=latent_dim)(inp)
        x = BatchNormalization()(x)
        x = Reshape(target_shape=(4, 4, depth * 8))(x)

        # design the generator to upsample the image 4x
        x = Generator.__add_block(x, depth * 4, filter_size)
        x = Generator.__add_block(x, depth * 2, filter_size)
        x = Generator.__add_block(x, depth, filter_size)
        x = Generator.__add_block(x, depth, filter_size)

        # turn the output into a 3D tensor, an image with 3 channels
        x = Conv2D(3, kernel_size=5, padding='same', activation='tanh')(x)

        return Model(inputs=inp, outputs=x)

    @staticmethod
    def __add_block(x, filters, filter_size):
        x = Deconvolution2D(filters, filter_size, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(0.3)(x)
        return x

