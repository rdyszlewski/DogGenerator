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








