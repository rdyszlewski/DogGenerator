from PIL.Image import Image
from keras import Sequential, Input, Model
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from model import Discriminator2, Discriminator, Generator


class Trainer:

    def train(self, data, labels, input_shape):

        depth = 64
        epochs = 1
        # TODO: z labels zrobić wartości liczbowe, zakodować to zero jedynkowo
        # TODO: podzielić

        discriminator = self.__create_discriminator(input_shape, depth)
        generator = self.__create_generator(input_shape, depth) # TODO: tutaj jako wejście powinno być rozmiar szumu
        gan = self.__create_gan(discriminator, generator)

        xTrain, xTest, yTrain, yTest = train_test_split(data, labels, test_size = 0.25, random_state=42)

        batch_size = 128

        for e in range(1, epochs + 1):
            print("Epoch %d" %e)
            for _ in tqdm(range(batch_size)):
                # generate random noise as an input to initialize the generator
                noise = np.random.normal(0, 1, [batch_size, 100]) # TODO: sprawdzić, dlaczego są takie wartości

                generated_images = generator.predict(noise)
                # get a random set of real images
                image_batch = xTrain[np.random.randint(low=0, high=xTrain.shape[0], size=batch_size)]
                # construct different batches of real and fakse data
                x = np.concatenate([image_batch, generated_images])

                # labels for generated and real data
                # TODO: sprawdzić, czy to na pewno będzie w ten spsoó”. MOżliwe, że przy większej liczbie etykiet będzie to wyglądało inaczej
                y = np.zeros(2*batch_size)
                y[:batch_size] = 0.9

                discriminator.trainable = True
                discriminator.train_on_batch(x, y)

                # tricking the noised input of the generator as real data
                noise = np.random.normal(0, 1, [batch_size, 100])
                y_gen = np.ones(batch_size)

                # training gan
                discriminator.trainable = False
                gan.train_on_batch(noise, y_gen) # TODO: pomyśleć nad tym. Co my tutaj przekazujemy
                if e == 1 or e % 20 == 0:
                    self.plot_generated_images(e, generator)






        # input_shape = (128, 128)
        # input_size = input_shape[0] * input_shape[1] * 3
        # output_size = 10000
        # noise_shape = (0,0)
        # zeros = np.zeros((output_size, input_size))
        # # TODO: prawdopodobnie będzie tzeba zmienić rozmiar w metodzie
        # discriminator_model = Discriminator2.create_model(input_shape, output_size)
        #
        # # TODO: sprawdzić, czy nie będzie trzeba
        # data, labels = shuffle(data, labels)
        #
        # self.__train_discriminator(discriminator_model, data, input_size, output_size, zeros)
        # self.__plot(discriminator_model, output_size, input_size, input_shape, zeros)
        # # trenowanie

    def __create_discriminator(self, input_shape, depth):
        discriminator_optimizer = RMSprop(lr=0.0008, clipvalue=1.0, decay=6e-8)
        discriminator = Discriminator.create_model(input_shape, depth)

        discriminator.compile(loss='binary_crossentropy', optimizer=discriminator_optimizer, metrics=['accuracy'])
        discriminator.summary()

        return discriminator



    def __create_generator(self, noise_shape, depth):
        generator_optimizer = Adam(lr=0.0002, beta_1=0.5)
        generator = Generator.create_model(noise_shape, depth)
        generator.compile(loss='binary_crossentropy', optimizer=generator_optimizer)
        generator.summary()

        return generator


    def __create_gan(self, discriminator, generator):
        discriminator.trainable = False
        gan_input = Input(shape=(100,)) # TODO: dowiedzieć się, co to za wejście
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropu', optimizer='adam')
        return gan

    def plot_generated_images(self, epoch, generator, examples=100, dim=(10,10), figsize=(10,10)):
        noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100, 28, 28) # TODO: zobaczyć skąd ten rozmiar
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i+1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('gan_generated_image %d.png'%epoch)

    # def __train_discriminator(self,discriminator_model, data, input_size, output_size, zeros):
    #     print("Rozpoczynam trenowanie")
    #     train_y = (data[:output_size, :, :, :]/255.).reshape((-1, input_size))
    #     train_X = np.zeros((output_size, output_size))
    #     for i in range(output_size): train_X[i, i] = 1
    #
    #
    #     # trenowanie
    #     lr = 0.5
    #     for k in range(5):
    #         scheduler = LearningRateScheduler(lambda x: lr)
    #         h = discriminator_model.fit([zeros, train_X], train_y, epochs=100, batch_size = 64, callbacks=[scheduler], verbose=0)
    #         print('Epoch',(k+1)*10,'/50 - loss =',h.history['loss'][-1] )
    #         if h.history['loss'][-1] < 0.533: lr = 0.1
    #     del train_X, train_y, data # TODO: zastanowić się, czy ddane też usuwać
    #
    # def __plot(self, descriminator_model, output_size, input_size, input_shape, zeros):
    #     print("Rozpoczynam wyświetlanie")
    #     for k in range(5):
    #         plt.figure(figsize=(15,3))
    #         for j in range(5):
    #             xx = np.zeros((output_size))
    #             xx[np.random.randint(output_size)] = 1
    #             plt.subplot(1, 5, j+1)
    #             img = descriminator_model.predict([zeros[0,:].reshape((-1, input_size)), xx.reshape((-1, output_size))]).reshape((-1, input_shape[0], input_shape[1], 3))
    #             img = Image.fromarray((255*img).astype('uint8').reshape((input_shape[0], input_shape[1],3 )))
    #             plt.axis('off')
    #             plt.imshow(img)
    #         plt.show()
