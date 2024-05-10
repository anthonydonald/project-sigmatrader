from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class Metric:
    calc: Callable[
        [pd.DataFrame, Any],
        float | int,
    ]

    def report(self, prefix, *args, fmt="") -> str:
        return f"{prefix}{self.calc(*args):{fmt}}"


@dataclass
class PctMetric(Metric):

    def report(self, prefix, *args, fmt="5.2f") -> str:
        return f"{prefix}{self.calc(*args)*100:{fmt}}%"


def daily_returns(df: pd.DataFrame) -> pd.Series:
    daily_df = df.groupby([df.timestamp.dt.date]).last()
    return daily_df.value.pct_change(1)


def delta_days_hours_mins(delta: timedelta):
    return delta.days, delta.seconds // 3600, (delta.seconds // 60) % 60


def delta_days(delta: timedelta):
    days, hours, mins = delta_days_hours_mins(delta)
    return days + hours / 24 + mins / 1440


class TradeMetrics:
    # Percentage metrics
    total_return = PctMetric(
        calc=lambda s: (s.iloc[-1] / s.iloc[0] - 1),
    )
    mkt_sharpe_ratio = PctMetric(
        calc=lambda x: (daily_returns(x).mean() / daily_returns(x).std()) * 365**0.5,
    )
    trader_return = PctMetric(
        calc=lambda s: (s.iloc[-1] / s.iloc[0] - 1),
    )
    sharpe_ratio = PctMetric(
        calc=lambda book_df: (
            daily_returns(book_df).mean() / daily_returns(book_df).std()
        )
        * 365**0.5,
    )

    # Other metrics
    holding_time_mean_days = Metric(
        calc=lambda trades_df: (
            delta_days((trades_df.closed_timestamp - trades_df.open_timestamp).mean())
        ),
    )
    number_of_trades = Metric(
        calc=lambda trades_df: (len(trades_df)),
    )
