from tensorflow.keras.optimizers import Adam


class OptimizerParser:

    @staticmethod
    def get_optimizer( configuration):
        name = configuration["optimizer"]
        switcher = {
            "adam": OptimizerParser._get_adam_optimizer
        }
        return switcher.get(name)(configuration)

    @staticmethod
    def _get_adam_optimizer(configuration):
        learning_rate = configuration["learning_rate"]
        beta = configuration["beta"]
        return Adam(lr=learning_rate, beta_1=beta)