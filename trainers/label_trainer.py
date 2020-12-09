import os
from abc import ABC

from trainers.config import TrainerConfig
from trainers.trainer import Trainer


class LabelTrainer(Trainer, ABC):

    def __init__(self):
        super().__init__()
        self.__data_list = []
        self.__labels_map = {}

    def train(self, data, labels, input_shape):
        self.__data_list = os.listdir(TrainerConfig.data_path)
        # TODO: tutaj jednak będzie trzeba zrobić to trochę inaczej, ponieważ
        discriminator, gan, generator = self._prepare_models(input_shape)
        self.__train(generator, discriminator, gan, TrainerConfig.epochs)
