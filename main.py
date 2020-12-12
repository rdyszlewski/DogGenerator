from data.saver import DataSaver
from trainers.config import TrainerConfig
from trainers.gan_trainer2 import GanTrainer2

data_path = "/media/roman/07765B7E452A5B73/Machine Learning/Dogs"
configuration_path ="configuration/configuration.yaml"

def main():
    # __create_batches()
    __train()

    # with open(configuration_path) as f:
    #     configuration = yaml.full_load(f)
    # model_builder = ModelBuilder()
    # print(configuration)
    # result = model_builder.build(configuration["model"]["discriminator"])


def __train():
    trainer = GanTrainer2()
    trainer.train(None, None, TrainerConfig.input_shape)

def __create_batches():
    DataSaver.save_data(data_path, TrainerConfig.data_path, TrainerConfig.input_shape, 1024)


main()