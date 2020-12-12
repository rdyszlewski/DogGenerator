from sklearn.utils import shuffle
import numpy as np
import gc

from data import saver
from data.loaders.loader import DataLoader
from trainers.data_splitter import DataSplitter


class DiscriminatorTrainer:

    def __init__(self, label_weight, config, loader: DataLoader):
        self.__label_weight = label_weight
        self.__batch_size = config["train"]["batch_size"]
        self.__noise_size = config["model"]["generator"]["input_shape"]
        self.__iterations = config["train"]["discriminator_iterations"]
        self.__prepared_data_path = config["dataset"]["prepared_data_path"]
        self._loader = loader

    def train(self, discriminator, generator):
        discriminator.trainable = True
        # x_train = self.__load_iteration_data(iteration)
        x_train, labels = self._loader.get_next()
        for i in range(self.__iterations):
            x_train = shuffle(x_train)
            train_data, generated_images = self.__prepare_data_for_discriminator(generator, x_train)
            # positive
            discriminator.train_on_batch(train_data, np.array([self.__label_weight] * train_data.shape[0]))
            # negative
            discriminator.train_on_batch(generated_images, np.zeros(train_data.shape[0]))
            del train_data, generated_images
        del x_train
        gc.collect()

    def __prepare_data_for_discriminator(self, generator, x_train):
        noise = np.random.normal(0, 1, [self.__batch_size, self.__noise_size])
        generated_images = generator.predict(noise)
        train_data = DataSplitter.split(x_train, self._loader.get_iteration(), self.__batch_size)
        return train_data, generated_images
