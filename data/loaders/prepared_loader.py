import math

from sklearn.utils import shuffle
from data.loaders.loader import DataLoader
import os
import pickle

class PreparedDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__()
        data_path = config["dataset"]["prepared_data_path"]
        self.__batch_size = config["train"]["batch_size"]
        prepared_data_size = config["dataset"]["prepared_data_size"]
        files_names = os.listdir(data_path)
        self.__data_list = []
        for name in files_names:
            self.__data_list.append(data_path + '/' + name)
        self.__batch_in_data = math.floor(prepared_data_size / self.__batch_size)
        # self.__batch_in_data = 1
        self.__data_iterator = -1
        self.__current_data = None
        self.__current_labels = None


    def get_next(self):
        if self.__data_iterator == self.__batch_in_data or self.__data_iterator < 0:
            self.__data_iterator=0
            self.__current_data, self.__current_labels= self._load_next_data()
        start_index = self.__data_iterator * self.__batch_size
        end_index = start_index + self.__batch_size
        x = self.__current_data[start_index: end_index]
        labels = self.__current_data[start_index: end_index]
        x, labels = shuffle(x, labels)
        self.__data_iterator += 1
        return x, labels

    def _load_next_data(self):
        self._iteration += 1
        path = self.__data_list[self._iteration]
        data = pickle.load(open(path, "rb"))
        x_train, labels = shuffle(data[0], data[1])
        return x_train, labels

    def get_size(self):

        return len(self.__data_list) * self.__batch_in_data

    def next_epoch(self):
        self._iteration = -1
        self.__data_iterator = -1
        self.__data_list = shuffle(self.__data_list)
