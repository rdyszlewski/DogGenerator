from tensorflow.keras import Input, Model


class Gan:

    @staticmethod
    def create_model(discriminator, generator, configuration):
        input_size = configuration["model"]["generator"]["input_shape"]
        discriminator.trainable = False
        gan_input = Input(shape=(input_size,))
        x = generator(gan_input)
        gan_output = discriminator(x)
        gan = Model(inputs=gan_input, outputs=gan_output)
        return gan
