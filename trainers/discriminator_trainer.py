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

        self._positive_labels = np.array([self.__label_weight] * self.__batch_size)
        self._negative_labels = np.zeros(self.__batch_size)

    def train(self, discriminator, generator):
        discriminator.trainable = True
        x_train, labels = self._loader.get_next()
        for i in range(self.__iterations):
            x_train = shuffle(x_train)
            generated_images = self.__prepare_data_for_discriminator(generator)
            # positive
            discriminator.train_on_batch(x_train, self._positive_labels)
            # negative
            discriminator.train_on_batch(generated_images, self._negative_labels)
            del generated_images
        del x_train
        gc.collect()

    def __prepare_data_for_discriminator(self, generator):
        noise = np.random.normal(0, 1, [self.__batch_size, self.__noise_size])
        generated_images = generator.predict(noise)
        # train_data = DataSplitter.split(x_train, self._loader.get_iteration(), self.__batch_size)
        # return train_data, generated_images
        return generated_images
