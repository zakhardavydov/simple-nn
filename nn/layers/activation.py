import numpy as np

from .layer import Layer


class ActivationLayer(Layer):
    """
    Layer that is stacked to run the output of the neuron via the activation function
    """

    def __init__(self, activation):
        super(ActivationLayer, self).__init__()

        self.activation = activation

    def forward(self, x: np.array) -> np.array:
        """
        :param x:
        :return: activation function applied on x
        """
        self.input = x
        return self.activation.fun(x)

    def backprop(self, de_dy: np.array, lr: float) -> np.array:
        """
        Element product between the error derivative
        in relation to the output and the derivative of the activation function
        """
        return de_dy * self.activation.derivative(self.input)
