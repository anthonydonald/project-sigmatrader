import logging
from abc import ABC, abstractmethod

import pandas as pd

logger = logging.getLogger(__name__)


class DataSplitter(ABC):
    @abstractmethod
    def split(self, data): ...


class RelativeSplitter(DataSplitter):
    def __init__(self, **rel_splits: dict[str, float]):
        super().__init__()
        assert sum(rel_splits.values()) == 1
        self._rel_splits = pd.Series(rel_splits)

    def split(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        df_splits_map = {}
        cum_sum_splits = pd.Series(self._rel_splits).cumsum()

        end_idxs = (cum_sum_splits * len(data)).astype(int)
        start_idxs = end_idxs.shift(1, fill_value=0)
        end_idxs.iloc[-1] += 1

        start_end_idxs = list(zip(start_idxs, end_idxs))
        logger.debug("Splitter start, end indices: %s", start_end_idxs)

        for name, (s, e) in zip(self._rel_splits.keys(), start_end_idxs):
            df_splits_map[name] = (df_split := data.iloc[s:e])
            logger.info("Data split for %s shape: %s", name, df_split.shape)
        return df_splits_map
