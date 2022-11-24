from typing import List

import numpy as np

from .layers import Layer


class NN:

    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def layer(self, layer: Layer):
        self.layers.append(layer)

    def run(self, x: np.array) -> np.array:
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def _update_weight(self, predicted: np.array, gt: np.array, lr: float) -> float:
        de_dy = self.loss.derivative(predicted, gt)
        for layer in reversed(self.layers):
            de_dy = layer.backprop(de_dy, lr)
        return de_dy

    def fit(self, x: List[np.array], y: List[np.array], epochs: int, lr: float):
        for epoch in range(0, epochs):
            cost = 0
            for sample, y_true in zip(x, y):
                y_predicted = self.run(sample)
                self._update_weight(y_predicted, y_true, lr)
                cost += self.loss.fun(y_predicted, y_true)
            print(f"Current epoch: {epoch}")
            print(f"Cost: {cost}")
            print(f"Loss per sample {cost / len(x)}")
