import numpy as np


class Layer:
    """
    Generic class for any kind of layer that can be stacked in the neural net
    """

    def __init__(self):
        self.input = None

    def forward(self, x: np.array) -> np.array:
        pass

    def backprop(self, de_dy: np.array, lr: float) -> np.array:
        pass
