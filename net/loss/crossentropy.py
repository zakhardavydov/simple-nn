import numpy as np

from .loss import Loss


class CrossEntropyLoss(Loss):

    def fun(self, y_predicted: np.array, y_true: np.array) -> np.array:
        return -np.sum(y_true * np.log(y_predicted))

    def derivative(self, y_predicted: np.array, y_true: np.array) -> np.array:
        return -(y_true / y_predicted)
