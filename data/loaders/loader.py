
import abc
import os

class DataLoader:

    def __init__(self):
        self._iteration = -1


    @abc.abstractmethod
    def get_next(self):
        pass

    @abc.abstractmethod
    def get_size(self):
        pass

    @abc.abstractmethod
    def next_epoch(self):
        pass

    def get_iteration(self):
        return self._iteration