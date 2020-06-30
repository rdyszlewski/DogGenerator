import gc
import os

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm

from data.saver import DataLoader
from plot import Plot
from trainers.config import TrainerConfig
from trainers.trainer import Trainer


class GanTrainer2(Trainer):

    def __init__(self):
        super().__init__()
        self.__data_list = []

    def train(self, data, labels, input_shape):

        self.__data_list = os.listdir(TrainerConfig.data_path)
        discriminator, gan, generator = self._prepare_models(input_shape)
        self.__train(generator, discriminator, gan, TrainerConfig.epochs)

    def __train(self, generator, discriminator, gan, epochs):
        for e in range(epochs):
            print("Epoch %d" % e)
            iteration = 0
            self.__train_epoch(discriminator, gan, generator, iteration)
            Plot.plot_generated_images(e, generator)

    def __train_epoch(self, discriminator, gan, generator, iteration):
        self.__data_list = shuffle(self.__data_list)

        for _ in tqdm(range(len(self.__data_list))):
            x_train = self.__load_iteration_data(iteration)
            self.__train_discriminator(discriminator, generator, x_train, iteration)
            self._train_gan(gan, discriminator, iteration)
            iteration += 1
            del x_train
            gc.collect()

    def __load_iteration_data(self, iteration):
        path = TrainerConfig.data_path + '/' + self.__data_list[iteration]
        x_train, labels = DataLoader.load_data(path)
        return x_train

    def __train_discriminator(self, discriminator, generator, x_train, iteration):
        discriminator_iterations = 5
        discriminator.trainable = True
        for i in range(discriminator_iterations):
            x_train = shuffle(x_train)
            x, y = self.__prepare_data_for_discriminator(generator, x_train, TrainerConfig.batch_size, iteration)
            discriminator.train_on_batch(x, y)
            del x, y
        # discriminator.fit(x, y, batch_size= TrainerConfig.batch_size, verbose=1)

    def __prepare_data_for_discriminator(self, generator, x_train, batch_size, iteration):
        noise = np.random.normal(0, 1, [batch_size, TrainerConfig.noise_size])
        generated_images = generator.predict(noise)
        # real_images = x_train[np.random.randint(low=0, high=x_train.shape[0], size=examples)]
        train_data = self.__split_train_data(x_train, iteration, batch_size)
        x = np.concatenate([train_data, generated_images])
        y = self._get_y_for_discriminator_train(train_data.shape[0])
        x, y = shuffle(x, y)
        return x, y

    def __split_train_data(self, x_train, iteration, batch_size):
        start_index = batch_size * iteration
        end_index = batch_size * iteration + batch_size
        if end_index >= x_train.shape[0]:
            end_index = x_train.shape[0] - 1
        train_data = x_train[start_index: end_index]
        if len(train_data) > batch_size:
            remains = train_data.shape[0] - len(train_data)
            train_data.concatenate(train_data[:remains - 1])
        return train_data

    def _get_y_for_discriminator_train(self, batch_size):
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1
        return y

    def _train_gan(self, gan, discriminator, iteration):
        noise, y_gen = self._prepare_data_for_gan_training(TrainerConfig.batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)
        del noise, y_gen

    def _prepare_data_for_gan_training(self, examples):
        # tricking the noised input of the generator as real data
        noise = np.random.normal(0, 1, [examples, TrainerConfig.noise_size])
        y_gen = np.ones(examples)
        y_gen = np.array(y_gen)
        return noise, y_gen
