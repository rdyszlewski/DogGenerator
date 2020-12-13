from sklearn.utils import shuffle
import numpy as np
import gc

from data.loaders.loader import DataLoader
from trainers.data_splitter import DataSplitter


class DiscriminatorTrainer:

    def __init__(self, config, loader: DataLoader):
        self.__label_weight = config["train"]["discriminator"]["labels_weight"]
        self.__batch_size = config["train"]["batch_size"]
        self.__noise_size = config["model"]["generator"]["input_shape"]
        self.__iterations = config["train"]["discriminator"]["iterations"]
        self.__prepared_data_path = config["dataset"]["prepared_data_path"]
        self._loader = loader

        self._positive_labels = np.array([self.__label_weight] * self.__batch_size)
        self._negative_labels = np.zeros(self.__batch_size)

    def train(self, discriminator, generator):
        x_train, labels = self._loader.get_next()
        for i in range(self.__iterations):
            x_train = shuffle(x_train)
            generated_images = self.__prepare_data_for_discriminator(generator)
            # positive
            positive_labels = np.array([self.__label_weight] * self.__batch_size)
            result1 = discriminator.train_on_batch(x_train, positive_labels)
            # negative
            negative_labels = np.zeros(self.__batch_size)
            result = discriminator.train_on_batch(generated_images, negative_labels)
            del generated_images
        del x_train
        gc.collect()

    def __prepare_data_for_discriminator(self, generator):
        noise = np.random.normal(0, 1, [self.__batch_size, self.__noise_size])
        generated_images = generator.predict(noise)
        return generated_images
