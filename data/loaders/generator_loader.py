import random

from sklearn.utils import shuffle

from data.data import DataPreparator
from data.loaders.loader import DataLoader
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.builder.shape_parser import ShapeParser
import numpy as np

class GeneratorDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__()
        shape = ShapeParser.parse(config["model"]["discriminator"]["input_shape"])
        path = config["dataset"]["data_path"]
        self._batch_size = config["train"]["batch_size"]
        self._x_train, self._labels = DataPreparator.prepare_data(path, int(shape[0]), int(shape[1]))
        self._image_generator = ImageDataGenerator(rescale=1./255,
                                                   rotation_range=20,
                                                   width_shift_range=0.2,
                                                   height_shift_range=0.2,
                                                   zoom_range=0.2,
                                                   horizontal_flip=True,
                                                   fill_mode="nearest")

    def get_next(self):
        x, y = self.__prepare_elements()
        x_train =  self._image_generator.flow(x, y, self._batch_size)
        return x_train.x, x_train.y

    def __prepare_elements(self):
        x = []
        y = []
        for i in range(self._batch_size):
            index = random.randint(0, len(self._x_train)-1)
            x.append(self._x_train[index])
            y.append(self._labels[index])
        return np.array(x), np.array(y)

    def get_size(self):
        # return int(self._x_train.shape[0] / self._batch_size)
        return 20

    def next_epoch(self):
        self._x_train, self._labels = shuffle(self._x_train, self._labels)