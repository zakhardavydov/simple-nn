from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class Utils:

    @staticmethod
    def train_test_split(df: pd.DataFrame, split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        :param df: dataframe to split
        :param split: ration between train and test sets
        :return:
        """
        r = np.random.rand(len(df)) < split
        return df[r], df[~r]

    @staticmethod
    def clean_x(df: pd.DataFrame) -> List[np.array]:
        """
        :param df: source dataframe
        :return: list of input vectors
        """
        return [np.array([row]) for row in df[["x0", "x1", "x2", "x3"]].values.tolist()]

    @staticmethod
    def clean_y(rows: pd.Series, label2int: Dict[str, List[str]]) -> List[np.array]:
        """
        :param rows: raw series with labels
        :param label2int: map to use to convert str label into int label
        :return: cleaned labels
        """
        return [label2int[row] for row in rows]

    @staticmethod
    def get_index_label(raw: np.array):
        """
        :param raw: the output of the model
        :return: the index of the most-likely class
        """
        return raw.argmax(axis=0)

    @staticmethod
    def get_confusion_matrix(predicted: List[int], y_true: List[int], label_count: int) -> np.array:
        """
        Construct confusion matrix out of raw arrays
        :param predicted: array of model predictions
        :param y_true: array of ground truth
        :param label_count: how many labels we have in the dataset
        :return:
        """
        out = np.zeros([label_count, label_count])
        for p, t in zip(predicted, y_true):
            out[t][p] += 1
        return out

    @staticmethod
    def accuracy(cm: np.array) -> float:
        """
        Get the accuracy from the confusion matrix
        :param cm: confusion matrix
        :return: accuracy
        """
        total_samples = np.sum(cm)
        tp = 0
        for i in range(cm.shape[0]):
            tp += cm[i, i]
        return tp / total_samples

    @staticmethod
    def precision(cm: np.array):
        """
        Get the precision
        :param cm: confusion matrix
        :return: by-class precision
        """
        return np.diag(cm) / np.sum(cm, axis=1)

    @staticmethod
    def recall(cm: np.array):
        """
        Get the recall
        :param cm: confusion matrix
        :return: by-class recall
        """
        return np.diag(cm) / np.sum(cm, axis=0)

    @staticmethod
    def report(cm: np.array, int2label: Dict[int, str]) -> Tuple[float, np.array, np.array]:
        """
        Verbose report with by-class metrics
        :param cm: confusion matrix
        :param int2label: map to convert str label into int label
        :return: tuple with accuracy, precision & recall
        """
        accuracy = Utils.accuracy(cm)
        precision = Utils.precision(cm)
        recall = Utils.recall(cm)

        print(f"Accuracy: {accuracy}")
        for i in range(0, len(int2label)):
            print(f"[LABEL] {int2label[i]}:")
            print(f"|===| Precision: {precision[i]}")
            print(f"|===| Recall: {recall[i]}")

        return accuracy, precision, recall

    @staticmethod
    def plot_cm(cm: np.array, int2label: Dict[int, str]):
        """
        Map confusion matrix using seaborn
        :param cm: confusion matrix to plot
        :param int2label: map to convert str label into int label
        :return:
        """
        import seaborn as sn

        labels = list(int2label.values())
        df_cm = pd.DataFrame(
            cm,
            index=labels,
            columns=labels
        )
        plt.figure(figsize=(len(labels), len(labels)))
        sn.heatmap(df_cm, annot=True)
