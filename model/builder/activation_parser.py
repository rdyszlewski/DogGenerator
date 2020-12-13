from tensorflow.keras.layers import LeakyReLU, ReLU
from model.builder.shared import LayersPattern


class ActivationParser:

    def __init__(self):
        self._activation = None
        self._value = None
        pass

    def parse(self, value):
        activation = self._get_activation_creator(value)
        return activation(value)

    def _get_activation_creator(self, value):
        name = value.split("(")[0].lower()
        switcher = {
            "leakrelu": self.get_leaky_relu_activation,
            "relu": self.get_relu_activation
        }
        return switcher.get(name, self.get_relu_activation)

    def set_activation(self, value):
        self._value = value
        self._activation = self._get_activation_creator(value)

    def get_activation(self):
        return self._activation(self._value)

    def get_leaky_relu_activation(self, activation_value):
        value = LayersPattern.getParameters(activation_value)
        return LeakyReLU(value)

    def get_relu_activation(self, activation_value):
        return ReLU()