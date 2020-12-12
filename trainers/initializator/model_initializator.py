from tensorflow.keras.optimizers import Adam

from model.dcgan4 import Generator, Discriminator, Gan
from trainers.initializator.optimizer_parser import OptimizerParser


class ModelInitializator:

    @staticmethod
    def prepare_models(config):
        discriminator = Discriminator.create_model(config["model"]["discriminator"])
        discriminator.trainable = True
        generator = Generator.create_model(config["model"]["generator"])
        gan = Gan.create_model(discriminator, generator, config)

        ModelInitializator._init_model(discriminator, "discriminator", config)
        ModelInitializator._init_model(generator, "generator", config)
        ModelInitializator._init_model(gan, "gan", config)

        return generator, discriminator, gan

    @staticmethod
    def _init_model(model, model_name, config):
        configuration = config["train"][model_name]
        loss = configuration["loss"]
        optimizer = OptimizerParser.get_optimizer(configuration)
        model.compile(loss=loss, optimizer=optimizer)
        model.summary()


