from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.initializers import RandomNormal

from model.builder.model_builder import ModelBuilder


class Discriminator:
    @staticmethod
    def create_model(configuration):
        init = RandomNormal(mean=0.0, stddev=0.02)
        model_builder = ModelBuilder()
        model = model_builder.build(configuration)
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid', kernel_initializer=init))

        return model
