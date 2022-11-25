from typing import List, Dict

import numpy as np

from .layers import Layer


class NN:

    def __init__(self, loss):
        self.layers = []
        self.loss = loss

    def layer(self, layer: Layer):
        """
        Add layer to internal list of layers
        :param layer: layer to stack
        :return:
        """
        self.layers.append(layer)

    def run(self, x: np.array) -> np.array:
        """
        Run the model once by iterating
        the list of available layers and calling forward propagation on them
        Pump the result of the previous layer into the next one
        :param x: sample to run the model on
        :return: the final result
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def run_all(self, x: List[np.array]) -> List[np.array]:
        """
        Run model on the batch of samples
        :param x: batch of samples
        :return: model run on batch
        """
        return [self.run(sample) for sample in x]

    def _update_weight(self, predicted: np.array, gt: np.array, lr: float) -> float:
        """
        Iterate the layers back and call backprop method on the layer
        to update the weights and pump the derivative relative to the input of w + 1 layer
        to the backprop method of w layer
        :param predicted: results of forward pass
        :param gt: what value is expected
        :param lr: learning rate
        :return:
        """
        de_dy = self.loss.derivative(predicted, gt)
        for layer in reversed(self.layers):
            de_dy = layer.backprop(de_dy, lr)
        return de_dy

    def fit(
            self,
            x: List[np.array],
            y: List[np.array],
            epochs: int,
            lr: float,
            verbose: bool = True
    ) -> Dict[int, float]:
        """
        Let's fit (train) the model!!!
        :param x: predictors
        :param y: labels
        :param epochs: number of epochs to iterate
        :param lr: learning rate
        :param verbose: whether to print epoch stats
        :return: dict that maps epoch to loss at that epoch
        """
        loss_at_epoch = {}
        for epoch in range(0, epochs):
            cost = 0
            for sample, y_true in zip(x, y):
                y_predicted = self.run(sample)
                self._update_weight(y_predicted, y_true, lr)
                cost += self.loss.fun(y_predicted, y_true)
            loss_at_epoch[epoch] = cost
            if verbose:
                print(f"Current epoch: {epoch}")
                print(f"Cost: {cost}")
                print(f"Loss per sample {cost / len(x)}")
        return loss_at_epoch
