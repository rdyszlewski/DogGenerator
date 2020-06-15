import abc
import gc
import os

import numpy
import numpy as np
from keras import Input, Model, Sequential
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tqdm import tqdm

# from model import Discriminator, Generator
from model.model import Discriminator, Generator
from plot import Plot
from trainers.config import TrainerConfig


class Trainer:

    def __init__(self):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    @abc.abstractmethod
    def train(self, data, labels, input_shape):
        pass

    def _prepare_models(self, input_shape):
        discriminator = self._create_discriminator(input_shape, TrainerConfig.depth)
        generator = self._create_generator(input_shape,
                                           TrainerConfig.depth)  # TODO: tutaj jako wejście powinno być rozmiar szumu
        gan = self._create_gan(discriminator, generator)
        return discriminator, gan, generator

    def _prepare_data(self, data, labels):
        data, labels = shuffle(data, labels)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
        x_train, y_train = shuffle(x_train, y_train)
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5

        del data
        del labels
        gc.collect()
        return x_train, x_test, y_train, y_test

    def _create_discriminator(self, input_shape, depth):
        # discriminator_optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        # discriminator_optimizer = Adam(lr=0.0008, beta_1=0.5)
        optimizer = Adam(lr=0.0002)
        discriminator = Discriminator.create_model(input_shape, depth)

        # discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
        discriminator.trainable = True
        discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['mae'])
        discriminator.summary()

        return discriminator

    def _create_generator(self, noise_shape, depth):
        generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        generator = Generator.create_model(noise_shape, depth)
        generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
        generator.summary()

        return generator

    # def _create_gan(self, discriminator, generator):
    #     discriminator.trainable = False
    #     gan_input = Input(shape=(100,))  # TODO: dowiedzieć się, co to za wejście
    #     x = generator(gan_input)
    #     gan_output = discriminator(x)
    #     gan = Model(inputs=gan_input, outputs=gan_output)
    #     gan.compile(loss='binary_crossentropy', optimizer='adam')
    #     return gan

    def _create_gan(self, discriminator, generator):
        gan = Sequential()
        gan.add(generator)
        gan.add(discriminator)
        discriminator.trainable = False
        gan.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0002), metrics=['mae'])

        return gan

