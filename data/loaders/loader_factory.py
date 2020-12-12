import string

from data.loaders.generator_loader import GeneratorDataLoader
from data.loaders.prepared_loader import PreparedDataLoader


class LoaderFactory:

    @staticmethod
    def get_loader(loader_name: string, config):
        switcher = {
            "prepared": PreparedDataLoader(config),
            "batch": PreparedDataLoader(config),
            "generator": GeneratorDataLoader(config)
        }
        return switcher.get(loader_name)