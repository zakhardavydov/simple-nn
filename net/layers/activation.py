import numpy as np

from .layer import Layer


class ActivationLayer(Layer):

    def __init__(self, activation):
        super(ActivationLayer, self).__init__()

        self.activation = activation

    def forward(self, x: np.array) -> np.array:
        self.input = x
        return self.activation.fun(x)

    def backprop(self, de_dy: np.array, lr: float) -> np.array:
        return de_dy * self.activation.derivative(self.input)
