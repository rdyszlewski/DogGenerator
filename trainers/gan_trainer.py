from sklearn.utils import shuffle

from plot import Plot
from trainers.config import TrainerConfig
from trainers.trainer import Trainer
import numpy as np

class GanTrainer(Trainer):

    def __init__(self):
        super().__init__()

    def train(self, data, labels, input_shape):
        discriminator, gan, generator = self._prepare_models(input_shape)
        x_train, x_test, y_train, y_test = self._prepare_data(data, labels)
        self.__train(generator, discriminator, gan, x_train, TrainerConfig.epochs)

    def __train(self, generator, discriminator, gan, x_train, epochs):
        discriminator_examples = 200
        generator_examples = 200
        for e in range(epochs):
            print("Epoch %d" % e)
            self.__train_discriminator(discriminator, generator, x_train, discriminator_examples)
            self._train_gan(gan, discriminator, generator_examples)
            Plot.plot_generated_images(e, generator)


    def __train_discriminator(self, discriminator, generator, x_train, examples):
        discriminator.trainable = True
        for i in range(examples):
            x, y = self.__prepare_data_for_discriminator(generator, x_train, TrainerConfig.batch_size)
            discriminator.train_on_batch(x, y)
            # discriminator.fit(x, y, batch_size= TrainerConfig.batch_size, verbose=1)

    def __prepare_data_for_discriminator(self, generator, x_train, examples):
        noise = np.random.normal(0,1, [examples,100])
        generated_images = generator.predict(noise)
        real_images = x_train[np.random.randint(low=0, high=x_train.shape[0], size=examples)]
        x = np.concatenate([real_images, generated_images])
        y = self._get_y_for_discriminator_train(examples)
        x, y = shuffle(x, y)
        return x, y

    def _get_y_for_discriminator_train(self, batch_size):
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1
        return y

    def _train_gan(self, gan, discriminator, examples):
        for i in range(examples):
            noise, y_gen = self._prepare_data_for_gan_training(TrainerConfig.batch_size)
            discriminator.trainable = False
            # gan.train_on_batch(noise, y_gen)
            # gan.fit(noise, y_gen, batch_size=TrainerConfig.batch_size, verbose=1)
            gan.train_on_batch(noise, y_gen)

    def _prepare_data_for_gan_training(self, examples):
        # tricking the noised input of the generator as real data
        noise = np.random.normal(0, 1, [examples, 100])
        y_gen = np.ones(examples)
        y_gen = np.array(y_gen)
        return noise, y_gen