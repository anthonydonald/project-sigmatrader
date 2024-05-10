import logging
from abc import ABC, abstractmethod
from datetime import datetime

import gymnasium as gym
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from sigmatrader.data.tadata import get_ta_data

logger = logging.getLogger(__name__)


class Transformer(ABC):
    """TODO: summary

    Args:
        ABC (_type_): _description_
    """

    @abstractmethod
    def fit(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def transform(self, data: pd.DataFrame):
        pass

    def fit_transform(self, data: pd.DataFrame):
        self.fit(data)
        return self.transform(data)

    def bulk_transform(self, named_datasets: dict[str, pd.DataFrame]):
        transformed_datasets = {}
        for name, data in named_datasets.items():
            transformed_datasets[name] = self.transform(data)
        return transformed_datasets


class FeatureBuilder(Transformer):
    pass


class Filter(Transformer):
    pass


class MultiTransformer(Transformer):
    def __init__(self, transformers: list[Transformer]):
        super().__init__()
        self.transformers = transformers

    def fit(self, data: pd.DataFrame):
        for t in self.transformers:
            t.fit(data)
        return self

    def transform(self, data: pd.DataFrame):
        for t in self.transformers:
            data = t.transform(data)
        return data

    def fit_transform(self, data: pd.DataFrame):
        for t in self.transformers:
            data = t.fit_transform(data)

        return data


class TABuilder(FeatureBuilder):
    def __init__(self, data_interval="1d", **kwargs):
        super().__init__()
        period, term = int(data_interval[:-1]), data_interval[-1]
        self.period_multiplier = 24 // period if term == "h" else 1
        self.mkt_data_history = None
        self.ta_kwargs = kwargs

    def fit(self, data: pd.DataFrame):
        self.mkt_data_history = data
        return self

    def transform(self, data: pd.DataFrame):
        data_and_history = data
        history_index = self.mkt_data_history.index
        is_history_valid = (
            lambda df: len(data.index.intersection(self.mkt_data_history.index)) == 0
            and history_index[-1] == data.index[0] - 1
        )
        if is_history_valid(data):
            data_and_history = pd.concat(
                [self.mkt_data_history, data], verify_integrity=True
            )
        else:
            logging.debug("Data loss expected due to calculating technical indicators")
        ta_df = get_ta_data(
            df=data_and_history,
            period_multiplier=self.period_multiplier,
            **self.ta_kwargs,
        )
        feature_df = data.join(ta_df).dropna().reset_index(drop=True)
        return feature_df
