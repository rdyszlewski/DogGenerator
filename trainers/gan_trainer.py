import numpy as np

class GanTrainer:

    def __init__(self, config):
        self.__config = config
        self.__batch_size = self.__config["train"]["batch_size"]
        self.__noise_size = self.__config["model"]["generator"]["input_shape"]

    def train(self, gan, discriminator):
        noise, y_gen = self._prepare_data_for_gan_training()
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)
        del noise, y_gen

    def _prepare_data_for_gan_training(self):
        # tricking the noised input of the generator as real data
        noise = np.random.normal(0, 1, [self.__batch_size, self.__noise_size])
        y_gen = np.ones(self.__batch_size)
        y_gen = np.array(y_gen)
        return noise, y_gen
