from sklearn.utils import shuffle

from data.data import DataPreparator
from data.loaders.loader import DataLoader
from model.builder.shape_parser import ShapeParser


class BatchDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__()
        shape = ShapeParser.parse(config["model"]["discriminator"]["input_shape"])
        path = config["dataset"]["data_path"]
        self._batch_size = config["train"]["batch_size"]
        self._x_train, self._labels = DataPreparator.prepare_data(path, int(shape[0]), int(shape[1]))
        self._files_count = self._x_train.shape[0]
        self._batch_count = int(self._files_count / self._batch_size)


    def get_next(self):
        self._iteration += 1
        return self.__prepare_data()

    def __prepare_data(self):
        start_index = self._batch_size * self._iteration
        end_index = start_index + self._batch_size
        if end_index >= self._files_count:
            end_index = self._files_count - 1
        train_data = self._x_train[start_index: end_index]
        labels_data = self._labels[start_index: end_index]
        if len(train_data) < self._batch_size:
            remains = self._batch_size - len(train_data)
            train_data.concatendate(self._x_train[:remains-1])
            labels_data.concatendate(self._labels[:remains-1])
        return train_data, labels_data


    def get_size(self):
        return self._batch_count

    def next_epoch(self):
        self._iteration = -1
        self._x_train, self._labels = shuffle(self._x_train, self._labels)