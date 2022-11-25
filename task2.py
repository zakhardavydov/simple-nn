import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from nn import NN
from nn.utils import Utils
from nn.loss import CrossEntropyLoss, MSELoss
from nn.activation import LogSigActivation, TanhActivation, ReLUActivation
from nn.layers import FullyConnectedLayer, ActivationLayer


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
    """
    Load and clean the data from path
    :param path: path to load
    :return: cleaned train/test lists
    """
    df = pd.read_csv(path)
    df = df.sample(frac=1)

    train, test = Utils.train_test_split(df, 0.7)

    x_train = Utils.clean_x(train)
    x_test = Utils.clean_x(test)

    y_train = Utils.clean_y(train["y"], label2int)
    y_test = Utils.clean_y(test["y"], label2int)

    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")

    return x_train, y_train, x_test, y_test


if __name__ == "__main__":

    label_count = 3
    show_loss = True
    show_cm = True

    # Freeze random for the consistency
    np.random.seed(42)

    # Load the data
    x_train, y_train, x_test, y_test = load_data("./IrisData.csv")

    # Init the network
    nn = NN(CrossEntropyLoss())

    # Stack some layers
    nn.layer(FullyConnectedLayer(4, 5))
    nn.layer(ActivationLayer(LogSigActivation()))

    nn.layer(FullyConnectedLayer(5, 3))
    nn.layer(ActivationLayer(LogSigActivation()))

    nn.layer(FullyConnectedLayer(3, 3))
    nn.layer(ActivationLayer(LogSigActivation()))

    # Train the net
    loss_at_epoch = nn.fit(x_train, y_train, 5000, 0.01)

    # Run on test set
    predicted = [Utils.get_index_label(p[0]) for p in nn.run_all(x_test)]
    test_labels = [Utils.get_index_label(y) for y in y_test]

    # Get the confusion matrix
    cm = Utils.get_confusion_matrix(predicted, test_labels, label_count)

    # Generate some metrics
    accuracy, precision, recall = Utils.report(cm, int2label)

    if show_loss:
        # Visualize loss at epoch
        plt.plot(
            list(loss_at_epoch.keys()),
            list(loss_at_epoch.values()),
            linestyle='-',
            marker='o'
        )

    if show_cm:
        # Visualize confusion matrix
        Utils.plot_cm(cm, int2label)

    plt.show()
