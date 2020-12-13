import click
import yaml
import os
from data.saver import DataSaver
from model.builder.shape_parser import ShapeParser
from trainers.trainer import Trainer


@click.group()
@click.option('--config_path')
def train(config_path: str):
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    with open(config_path) as file:
        configuration = yaml.safe_load(file)
    trainer = Trainer(configuration)
    trainer.train()


@click.group()
@click.option('--input_path')
@click.option('--output_path')
@click.option('--image-shape', "for example (64,64,3)")
@click.option('--size')
def prepare_data(input_path: str, output_path: str, input_shape: str, size: int):
    shape = ShapeParser.parse(input_shape)
    DataSaver.save_data(input_path, output_path, shape, size)