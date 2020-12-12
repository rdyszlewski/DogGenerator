from sklearn.utils import shuffle
from tqdm import tqdm

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
        batch_count = x_train.shape[0] / TrainerConfig.batch_size
        print(batch_count)
        for e in range(epochs):
            print("Epoch %d" % e)
            iteration = 0
            for _ in tqdm(range(int(batch_count))):
                self.__train_discriminator(discriminator, generator, x_train, iteration)
                self._train_gan(gan, discriminator, iteration)
                iteration += 1
            Plot.plot_generated_images(e, generator)
            x_train = shuffle(x_train)




    def __train_discriminator(self, discriminator, generator, x_train, iteration):
        for i in range(2):
            discriminator.trainable = True
            x_train = shuffle(x_train)
            x, y = self.__prepare_data_for_discriminator(generator, x_train, TrainerConfig.batch_size, iteration)
            discriminator.train_on_batch(x, y)
        # discriminator.fit(x, y, batch_size= TrainerConfig.batch_size, verbose=1)

    def __prepare_data_for_discriminator(self, generator, x_train, batch_size, iteration):
        noise = np.random.normal(0,1, [batch_size,100])
        generated_images = generator.predict(noise)
        # real_images = x_train[np.random.randint(low=0, high=x_train.shape[0], size=examples)]

        start_index = batch_size * iteration
        end_index = batch_size * iteration + batch_size
        if end_index >= x_train.shape[0]:
            end_index = x_train.shape[0] -1
        train_data = x_train[start_index: end_index]
        if len(train_data) > batch_size:
            remains = train_data.shape[0]-len(train_data)
            train_data.concatenate(train_data[:remains - 1]) # TODO: sprawdzić to
        real_images = train_data
        if(real_images.shape[0] != batch_size):
            print("Wysąpił jakiś błąd")
        x = np.concatenate([real_images, generated_images])
        y = self._get_y_for_discriminator_train(real_images.shape[0])
        x, y = shuffle(x, y)
        return x, y

    def _get_y_for_discriminator_train(self, batch_size):
        y = np.zeros(2 * batch_size)
        y[:batch_size] = 1
        return y

    def _train_gan(self, gan, discriminator, iteration):

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