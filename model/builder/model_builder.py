import re

from tensorflow.keras.layers import Dropout
from tensorflow.keras import Sequential
from tensorflow.python.keras.layers import BatchNormalization

from model.builder.activation_parser import ActivationParser
from model.builder.layer_parser import LayerParser
from model.builder.shape_parser import ShapeParser


class ModelBuilder:

    def __init__(self):
        self._model = Sequential()
        self._activation = None
        self._auto_add_layers = []
        self._dropout = -1
        self._kernel_size = 3
        self._depth = 1
        self._strides = -1
        self._layers = []
        self._input_dim = [0,0]

        self._layer_parser = None
        self._activation_parser = None

    def build(self, config):
        self._auto_add_layers = config["auto_add_layers"].split(",")
        if "strides" in config:
            self._strides = config["strides"]
        self._dropout = config["dropout"]
        self._kernel_size = config["kernel_size"]
        self._depth = config["depth"]
        self._input_dim = ShapeParser.parse(config["input_shape"])

        self._layer_parser = LayerParser(self._depth, self._kernel_size, self._strides, self._input_dim)
        self._activation_parser = ActivationParser()
        self._activation_parser.set_activation(config["activation"])
        model = self._parse_layers(config["layers"])
        return model

    def _parse_layers(self, layers_text):
        layers_values = layers_text.split(";")
        for value in layers_values:
            layer, name = self._layer_parser.parse(value)
            if layer:
                self._model.add(layer)
                if name in self._auto_add_layers:
                    if("conv" in layers_text):
                        self._model.add(BatchNormalization())
                    if self._activation:
                        self._model.add(self._activation_parser.get_activation())
                    if self._dropout > 0:
                        self._model.add(Dropout(self._dropout))
        return self._model




