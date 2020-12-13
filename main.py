from trainers.trainer import  Trainer
import yaml
import os

configuration_path ="configuration/configuration.yaml"

def main():
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    __train()


def __train():
    with open(configuration_path) as file:
        configuration = yaml.safe_load(file)
    trainer = Trainer(configuration)
    trainer.train()


main()