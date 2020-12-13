import numpy as np
import matplotlib.pyplot as plt

from model.builder.shape_parser import ShapeParser


class Plot:

    @staticmethod
    def plot_generated_images(epoch, generator, config, examples=25, dim=(5, 5), figsize=(20, 20)):
        # TODO: dodać rozmiary obrazków do ustawień
        noise_size = config["model"]["generator"]["input_shape"]
        input_shape = ShapeParser.parse(config["model"]["discriminator"]["input_shape"])
        noise = np.random.normal(loc=0, scale=1, size=[examples, noise_size])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(examples, *input_shape)

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