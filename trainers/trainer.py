from tqdm import tqdm

from data.loaders.batch_loader import BatchDataLoader
from data.loaders.loader import DataLoader
from data.loaders.loader_factory import LoaderFactory
from data.loaders.prepared_loader import PreparedDataLoader
from plot import Plot
from trainers.discriminator_trainer import DiscriminatorTrainer
from trainers.gan_trainer import GanTrainer
from trainers.initializator.model_initializator import ModelInitializator


class Trainer:

    def __init__(self, config):
        super().__init__()
        self._config = config
        self._discriminator_label_weight = config["train"]["discriminator_labels_weight"]
        # self._loader: DataLoader = PreparedDataLoader(config)
        self._loader: DataLoader = LoaderFactory.get_loader(config["train"]["loader"], config)
        self._discriminator_trainer: DiscriminatorTrainer = DiscriminatorTrainer(self._discriminator_label_weight, config, self._loader)
        self._gan_trainer: GanTrainer = GanTrainer(config)

    def train(self):
        # input_shape = ShapeParser.parse(self._config["model"]["discriminator"]["input_shape"])
        generator, discriminator, gan = ModelInitializator.prepare_models(self._config)
        self.__standard_train(generator, discriminator, gan, self._config)

    def __standard_train(self, generator, discriminator, gan, config):
        epochs = config["train"]["epochs"]
        save_result_interval = config["train"]["save_result_interval"]
        for epoch in range(epochs):
            print("Epoch %d" % epoch)
            self.__train_epoch(generator, discriminator, gan)
            if epochs % save_result_interval == 0:
                Plot.plot_generated_images(epoch, generator)

    def __train_epoch(self, generator, discriminator, gan):
        for _ in tqdm(range(self._loader.get_size()-1)):
            self._discriminator_trainer.train(discriminator, generator)
            self._gan_trainer.train(gan, discriminator)
            self._loader.next_epoch()

