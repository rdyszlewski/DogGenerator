import abc
import gc
import os

from keras import Input, Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from model.dcgan3 import Discriminator, Generator
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
                                           TrainerConfig.depth)
        gan = self._create_gan(discriminator, generator)
        return discriminator, gan, generator

    def _prepare_data(self, data, labels):
        data, labels = shuffle(data, labels)
        x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state=42)
        x_train, y_train = shuffle(x_train, y_train)
        x_train, x_test = self.__standarize_data(x_train, x_test)

        del data, labels
        gc.collect()
        return x_train, x_test, y_train, y_test

    def __standarize_data(self, x_train, x_test):
        x_train = (x_train - 127.5) / 127.5
        x_test = (x_test - 127.5) / 127.5
        return x_train, x_test

    def _create_discriminator(self, input_shape, depth):

        discriminator = Discriminator.create_model(input_shape, depth)
        discriminator.trainable = True
        # TODO: w tym momencie można zrobić kompilacje modelu
        return discriminator

    def _create_generator(self, noise_shape, depth):
        # TODO: przejrzeć się temu. To chyba nie jest noise_shape
        generator = Generator.create_model(noise_shape, depth, TrainerConfig.noise_size)
        return generator

    def _create_gan(self, discriminator, generator):
        discriminator.trainable = False
        gan_input = Input(shape=(100,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer = Adam(lr = 0.0002, beta_1 = 0.5))
        return gan

