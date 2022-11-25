import numpy as np

from .loss import Loss


class MSELoss(Loss):

    def fun(self, y_predicted: np.array, y_true: np.array) -> np.array:
        return np.mean(np.power(y_true - y_predicted, 2))

    def derivative(self, y_predicted: np.array, y_true: np.array) -> np.array:
        return 2 * (y_predicted - y_true) / y_predicted.size
