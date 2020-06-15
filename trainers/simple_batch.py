from sklearn.utils import shuffle

from plot import Plot
from trainers.config import TrainerConfig
from trainers.trainer import Trainer

import numpy as np
from tqdm import tqdm


class SimpleBatchTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def train(self, data, labels, input_shape):
        discriminator, gan, generator = self._prepare_models(input_shape)
        x_train, x_test, y_train, y_test = self._prepare_data(data, labels)
        self._train_with_batches(generator, discriminator, gan, x_train, TrainerConfig.batch_size,
                                 TrainerConfig.epochs)

    def _train_with_batches(self, generator, discriminator, gan, x_train, batch_size, epochs):
        for e in range(1, epochs + 1):
            print("Epoch %d" % e)
            for _ in tqdm(range(batch_size)):
                self._train_discriminator(discriminator, generator, x_train, batch_size)
                self._train_gan(gan, discriminator, batch_size)
            if e == 1 or e % 20 == 0:
                Plot.plot_generated_images(e, generator)

    def _train_discriminator(self, discriminator, generator, x_train, batch_size):
        x, y = self._prepare_data_for_discriminator_training(batch_size, generator, x_train)
        discriminator.trainable = True
        discriminator.train_on_batch(x, y)

    def _prepare_data_for_discriminator_training(self, batch_size, generator, x_train):
        # generate random noise to initialize generator
        noise = np.random.normal(0, 1, [batch_size, 100])  # TODO: sprawdzić, dlaczego są takie wartości
        generated_images = generator.predict(noise)
        real_images = x_train[np.random.randint(low=0, high=x_train.shape[0], size=batch_size)]
        x = np.concatenate([real_images, generated_images])
        y = self._get_y_for_discriminator_train(batch_size)
        x, y = shuffle(x, y)
        return x, y

    def _get_y_for_discriminator_train(self, batch_size):
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1
        return y

    def _prepare_data_for_gan_training(self, batch_size):
        # tricking the noised input of the generator as real data
        noise = np.random.normal(0, 1, [batch_size, 100])
        y_gen = np.ones(batch_size)
        y_gen = np.array(y_gen)
        noise, y_gen = shuffle(noise, y_gen)
        return noise, y_gen

    def _train_gan(self, gan, discriminator, batch_size):
        noise, y_gen = self._prepare_data_for_gan_training(batch_size)
        discriminator.trainable = False
        gan.train_on_batch(noise, y_gen)

