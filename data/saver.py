import pickle

from data.data import DataPreparator
import numpy as np

class DataSaver:

    @staticmethod
    def save_data(input_path, output_path, input_shape, batch_size):
        data, labels = DataPreparator.prepare_data(input_path, input_shape[0], input_shape[1])

        counter = 0
        data_to_save = []
        labels_to_save = []
        for i in range(len(data)):
            x = (data[i] - 127.5) / 127.5
            data_to_save.append(x)
            labels_to_save.append(labels[i])
            if len(data_to_save) == batch_size:
                path = output_path + "/" + 'data' + str(counter) + '.pck'
                DataSaver.__save_data(path,(np.array(data_to_save), np.array(labels_to_save)))
                counter += 1
                data_to_save.clear()
                labels_to_save.clear()

    @staticmethod
    def __save_data(output_path, data):
        pickle.dump(data, open(output_path, 'wb'))


