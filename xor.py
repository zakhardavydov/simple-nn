import numpy as np
import pandas as pd

from net import NN
from net.utils import Utils
from net.loss import MSELoss
from net.activation import TanhActivation
from net.layers import FullyConnectedLayer, ActivationLayer


if __name__ == "__main__":
    x_train = [np.array(x) for x in [[0, 0], [0, 1], [1, 0], [1, 1]]]
    y_train = [np.array(x) for x in [[0], [1], [1], [0]]]

    nn = NN(MSELoss())

    nn.layer(FullyConnectedLayer(2, 3))
    nn.layer(ActivationLayer(TanhActivation()))

    nn.layer(FullyConnectedLayer(3, 1))
    nn.layer(ActivationLayer(TanhActivation()))

    nn.fit(x_train, y_train, 1000, 0.0001)

    for sample, gt in zip(x_train, y_train):
        result = nn.run(sample)
        print(result)
