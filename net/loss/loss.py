import numpy as np


class Loss:

    def fun(self, y_predicted: np.array, y_true: np.array) -> np.array:
        pass

    def derivative(self, y_predicted: np.array, y_true: np.array) -> np.array:
        pass
