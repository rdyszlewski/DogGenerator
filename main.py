import numpy
from numpy import asarray

from data import DataPreparator
from trainer import Trainer

data_path = "/media/roman/07765B7E452A5B73/Machine Learning/Dogs"

def main():
    print("Rozpoczynam ładowanie danych")
    input_shape = (128, 128, 3)
    data, labels = DataPreparator.prepare_data(data_path, input_shape[0], input_shape[1])
    trainer = Trainer()
    trainer.train(data, labels, input_shape)
    print("Załadowano, zmieniam format")




main()