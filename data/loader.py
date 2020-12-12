import abc

class ILoader(abc.ABC):

    def __init__(self):
        # TODO: wstawić tutaj preparator
        pass

    @abc.abstractmethod
    def load(self, load_prepared:bool=False):
        pass