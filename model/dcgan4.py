from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras import Input
from tensorflow.keras.models import Model

from model.builder.model_builder import ModelBuilder


class Generator:

    @staticmethod
    def create_model(configuration):

        # model = Sequential()
        init = RandomNormal(mean=0.0, stddev=0.02)
        model_builder = ModelBuilder()
        model = model_builder.build(configuration)
        # for layer in layers:
        #     model.add(layer)

        # TODO: może można jakoś wrzucić to do buildera
        model.add(Conv2D(3, kernel_size=3, activation='tanh', padding='same', kernel_initializer=init))

        return model



class Discriminator:
    @staticmethod
    def create_model(configuration):
        init = RandomNormal(mean=0.0, stddev=0.02)
        # model = Sequential()
        model_builder = ModelBuilder()
        model = model_builder.build(configuration)
        # for layers in layers:
        #     model.add(layers)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

        return model



class Gan:

    @staticmethod
    def create_model(discriminator, generator, configuration):
        input_size = configuration["model"]["generator"]["input_shape"]
        discriminator.trainable = False
        gan_input = Input(shape=(input_size,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        return gan
