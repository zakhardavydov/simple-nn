import numpy as np

from .layer import Layer


class FullyConnectedLayer(Layer):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            use_adam: bool = False,
            bm: float = 0.9,
            bv: float = 0.999,
            e=1e-8
    ):
        super(FullyConnectedLayer, self).__init__()

        self.in_size = in_size
        self.out_size = out_size

        self.weight = np.random.rand(in_size, out_size) - 0.5
        self.bias = np.random.rand(1, out_size) - 0.5

        self.use_adam = use_adam

        self.e = e
        self.bm = bm
        self.bv = bv
        self.m = np.zeros(shape=(in_size, out_size))
        self.v = np.zeros(shape=(in_size, out_size))

    def forward(self, x: np.array) -> np.array:
        """
        Apply Y= WX + B
        :param x: raw input
        :return: output of the layer
        """
        self.input = x
        return self.bias + np.dot(self.input, self.weight)

    def adam(self, gradients, lr) -> np.array:
        """
        Attempt to do Adam optimizer
        :param gradients: raw gradients
        :param lr: learning rate
        :return:
        """
        self.m = self.bm * self.m + (1 - self.bm) * self.weight
        self.v = self.bv * self.v + (1 - self.bv) * np.power(self.weight, 2)
        return lr * self.m / (1 - np.power(self.bm, gradients)) / (np.sqrt(self.v / (1 - np.power(self.bv, gradients))) + self.e)

    def backprop(self, de_dy: np.array, lr: float) -> np.array:
        """
        Calculate the dEdx and return it. Update the weights (depending on the mode) alongside
        :param de_dy: error derivative in respect to the output of the layer
        :param lr: learning rate
        :return: error derivative in respect to the input of the layer
        """
        if self.use_adam:
            adam = self.adam(np.dot(self.input.T, de_dy), lr)
            self.weight -= adam
        else:
            self.weight -= np.dot(self.input.T, de_dy) * lr
        self.bias -= de_dy * lr
        return np.dot(de_dy, self.weight.T)
