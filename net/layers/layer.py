import numpy as np


class Layer:

    def __init__(self):
        self.input = None

    def forward(self, x: np.array) -> np.array:
        pass

    def backprop(self, de_dy: np.array, lr: float) -> np.array:
        pass
