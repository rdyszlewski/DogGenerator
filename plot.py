import numpy as np
import matplotlib.pyplot as plt

from trainers.config import TrainerConfig


class Plot:

    @staticmethod
    def plot_generated_images(epoch, generator, examples=25, dim=(5, 5), figsize=(20, 20)):
        # TODO: dodać rozmiary obrazków do ustawień
        noise = np.random.normal(loc=0, scale=1, size=[examples, TrainerConfig.noise_size])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(examples, *TrainerConfig.input_shape)

        generated_images = (generated_images + 1) / 2
        plt.figure(figsize=figsize)
        for i in range(generated_images.shape[0]):
            plt.subplot(dim[0], dim[1], i + 1)
            plt.imshow(generated_images[i], interpolation='nearest')
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(TrainerConfig.output_path % epoch)
        plt.close()
        del generated_images, noise