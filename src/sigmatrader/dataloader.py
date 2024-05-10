import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from sigmatrader.data.coindata import fetch_data

logger = logging.getLogger(__name__)


class DataLoader(ABC):
    """Abstract class for loading data"""

    @abstractmethod
    def load(self):
        pass


@dataclass
class BinanceLoader(DataLoader):
    symbol: str
    start_date: datetime
    end_date: datetime
    interval: str = "1d"

    def load(self) -> pd.DataFrame:
        data = fetch_data(
            pair=self.symbol,
            start_dt=self.start_date,
            end_dt=self.end_date,
            interval=self.interval,
        )
        return data
