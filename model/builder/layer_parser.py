from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import UpSampling2D, Dense
from tensorflow.keras.layers import Reshape

from model.builder.shared import LayersPattern

class LayerParser:

    def __init__(self, depth:int, kernel_size:int, strides: int, input_shape):
        self._depth: int = depth
        self._kernel_size: int = kernel_size
        self._strides: int = strides
        self._input_shape = input_shape

        self._first_layer = True
        self._initializator = RandomNormal(mean=0.0, stddev=0.02)

    def parse(self, value):
        name = value.split("(")[0].lower().strip().rstrip()
        switcher = {
            "conv": self.get_conv_layer,
            "upscaling": self.get_upscaling,
            'dense': self.get_dense,
            'reshape': self.get_reshape
        }
        if name not in switcher:
            return None, None
        return switcher.get(name)(value), name


    def get_conv_layer(self, value_text):
        FILTERS = 0
        KERNEL_SIZE = 1
        STRIDES = 2
        parameters = LayersPattern.getParameters(value_text).split(",")
        filters = int(parameters[FILTERS]) * self._depth if self._depth > 0 else int(parameters[FILTERS])
        kernel_size = int(parameters[KERNEL_SIZE]) if len(parameters) > KERNEL_SIZE else self._kernel_size
        strides = int(parameters[STRIDES]) if len(parameters) > STRIDES else self._strides

        if self._first_layer:
            self._first_layer = False
            layer = Conv2D(filters, kernel_size, padding="same", kernel_initializer= self._initializator, input_shape=self._input_shape)
        else:
            layer = Conv2D(filters, kernel_size, padding="same", kernel_initializer= self._initializator)

        if strides > 0:
            layer.strides = (self._strides, self._strides)
        return layer

    def get_upscaling(self, value_text):
        return UpSampling2D()

    def get_dense(self, value_text):
        value = int(LayersPattern.getParameters(value_text))
        value = self._depth * value if self._depth > 0 else value
        if self._first_layer:
            self._first_layer = False
            return Dense(value, kernel_initializer=self._initializator, input_dim=self._input_shape)
        return Dense(value, kernel_initializer=self._initializator)

    def get_reshape(self, reshape_text):
        value = LayersPattern.getParameters(reshape_text).split(",")
        depth = int(value[2])
        last_value = self._depth * depth if self._depth > 0 else depth
        shape = (int(value[0]), int(value[1]), last_value)
        return Reshape(shape)


