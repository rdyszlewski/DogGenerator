from sklearn.utils import shuffle
from data.loaders.loader import DataLoader
import os
import pickle

class PreparedDataLoader(DataLoader):

    def __init__(self, config):
        super().__init__()
        data_path = config["dataset"]["prepared_data_path"]
        files_names = os.listdir(data_path)
        self.__data_list = []
        for name in files_names:
            self.__data_list.append(data_path + '/' + name)


    def get_next(self):
        self._iteration += 1
        path =  self.__data_list[self._iteration]
        data = pickle.load(open(path, "rb"))
        x_train, labels = shuffle(data[0], data[1])
        return x_train, labels

    def get_size(self):
        return len(self.__data_list)

    def next_epoch(self):
        self._iteration = -1
        self.__data_list = shuffle(self.__data_list)
