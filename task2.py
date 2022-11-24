import numpy as np
import pandas as pd

from net import NN
from net.utils import Utils
from net.loss import CrossEntropyLoss, MSELoss
from net.activation import LogSigActivation, TanhActivation
from net.layers import FullyConnectedLayer, ActivationLayer


label2int = {
    "Iris-setosa": np.array([0.8, 0.1, 0.1]),
    "Iris-versicolor": np.array([0.1, 0.8, 0.1]),
    "Iris-virginica": np.array([0.1, 0.1, 0.8])
}

int2label = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
    2: "Iris-virginica"
}


def load_data(path: str):
    df = pd.read_csv(path)
    df = df.sample(frac=1)

    train, test = Utils.train_test_split(df, 0.8)
    x_train = Utils.clean_x(train)
    x_test = Utils.clean_x(test)

    y_train = Utils.clean_y(train["y"], label2int)
    y_test = Utils.clean_y(test["y"], label2int)

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = load_data("./IrisData.csv")

    print(x_train)
    print(y_train)

    nn = NN(MSELoss())

    nn.layer(FullyConnectedLayer(4, 5))
    nn.layer(ActivationLayer(LogSigActivation()))

    nn.layer(FullyConnectedLayer(5, 3))
    nn.layer(ActivationLayer(LogSigActivation()))

    nn.fit(x_train, y_train, 500, 0.01)

    for sample, gt in zip(x_test, y_test):
        result = nn.run(sample)
        print(result)
        gt_index = gt.argmax(axis=0)
        result_index = result[0].argmax(axis=0)
        if gt_index == result_index:
            print("CORRECT")
        else:
            print("INCORRECT")
