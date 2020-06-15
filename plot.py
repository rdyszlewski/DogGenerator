import numpy as np
import matplotlib.pyplot as plt

class Plot:

    @staticmethod
    def plot_generated_images(epoch, generator, examples=25, dim=(5, 5), figsize=(20, 20)):
        noise = np.random.normal(loc=0, scale=1, size=[examples, 100]) # TODO: tutaj jest rozmiar wejścia
        generated_images = generator.predict(noise)
        # generated_images = generated_images.reshape(100, 28, 28) # TODO: zobaczyć skąd ten rozmiar
        generated_images = generated_images.reshape(examples, 64, 64, 3)  # TODO: powprowadzać jakieś stałe
        # generated_images = (generated_images + 1) / 2
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig('output/gan_generated_image %d.png' % epoch)