import os
from sklearn.utils import shuffle
from tqdm import tqdm
from model.builder.shape_parser import ShapeParser
from plot import Plot
from trainers.config import TrainerConfig
from trainers.discriminator_trainer import DiscriminatorTrainer
from trainers.gan_trainer import GanTrainer
from trainers.trainer import Trainer


class GanTrainer2(Trainer):

    def __init__(self):
        super().__init__()
        self.__data_list = []
        self._discriminator_label_weight = None
        self._discriminator_trainer: DiscriminatorTrainer = None
        self._gan_trainer: GanTrainer = None

    def train(self, data, labels, config):
        self._discriminator_label_weight = config["train"]["discriminator_labels_weight"]
        input_shape = ShapeParser.parse(config["model"]["discriminator"]["input_shape"])
        self.__data_list = os.listdir(TrainerConfig.data_path)
        self._discriminator_trainer = DiscriminatorTrainer(self._discriminator_label_weight, self.__data_list, config)
        self._gan_trainer = GanTrainer(config)
        discriminator, gan, generator = self._prepare_models(input_shape)
        # TODO: zmienić to
        self.__standard_train(generator, discriminator, gan, config)

    def __standard_train(self, generator, discriminator, gan, config):
        epochs = config["train"]["epochs"]
        save_result_interval = config["train"]["save_result_interval"]
        for epoch in range(epochs):
            print("Epoch %d" % epoch)
            self.__train_epoch(generator, discriminator, gan, config)
            if epochs % save_result_interval == 0:
                Plot.plot_generated_images(epoch, generator)

    def __train_epoch(self, generator, discriminator, gan, config):
        self.__data_list = shuffle(self.__data_list)
        iteration = 0
        for _ in tqdm(range(len(self.__data_list))):
            self._discriminator_trainer.train(discriminator, generator, iteration)
            self._gan_trainer.train(gan, discriminator)
            iteration += 1
