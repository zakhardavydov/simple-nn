from typing import Tuple, List, Dict

import numpy as np
import pandas as pd


class Utils:

    @staticmethod
    def train_test_split(df: pd.DataFrame, split: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
        r = np.random.rand(len(df)) < split
        return df[r], df[~r]

    @staticmethod
    def clean_x(df: pd.DataFrame) -> List[np.array]:
        return [np.array([row]) for row in df[["x0", "x1", "x2", "x3"]].values.tolist()]

    @staticmethod
    def clean_y(rows: pd.Series, label2int: Dict[str, List[str]]) -> List[np.array]:
        return [label2int[row] for row in rows]
