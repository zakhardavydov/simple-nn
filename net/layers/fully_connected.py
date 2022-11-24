import numpy as np

from .layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(self, in_size: int, out_size: int):
        super(FullyConnectedLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = np.random.rand(in_size, out_size) - 0.5
        self.bias = np.random.rand(1, out_size) - 0.5

    def forward(self, x: np.array) -> np.array:
        self.input = x
        return self.bias + np.dot(self.input, self.weight)

    def backprop(self, de_dy: np.array, lr: float) -> np.array:
        de_dx = np.dot(de_dy, self.weight.T)
        weight_update = np.dot(self.input.T, de_dy)

        self.weight -= weight_update * lr
        self.bias -= de_dy * lr

        return de_dx
