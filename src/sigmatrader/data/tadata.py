from datetime import datetime

import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands


def get_ta_data(
    df,
    period_multiplier: int = 1,
    **kwargs,  # extend on this to make TA fields configurable
) -> pd.DataFrame:
    fast_sma_window = kwargs.get("fast_sma_window", 50)
    slow_sma_window = kwargs.get("fast_sma_window", 100)
    assert fast_sma_window <= slow_sma_window

    ta_df = pd.DataFrame([])

    boll_bands = BollingerBands(
        close=df["close"], window=20 * period_multiplier, window_dev=2
    )

    ta_df["bb_mavg"] = boll_bands.bollinger_mavg()
    ta_df["bb_hband"] = boll_bands.bollinger_hband()
    ta_df["bb_lband"] = boll_bands.bollinger_lband()

    sma_fast = SMAIndicator(
        close=df["close"], window=fast_sma_window * period_multiplier
    )
    sma_fast = SMAIndicator(
        close=df["close"], window=slow_sma_window * period_multiplier
    )

    ta_df[f"sma_{str(fast_sma_window)}"] = sma_fast.sma_indicator()
    ta_df[f"sma_{str(slow_sma_window)}"] = sma_fast.sma_indicator()

    macd = MACD(
        close=df["close"],
        window_slow=26 * period_multiplier,
        window_fast=12 * period_multiplier,
        window_sign=9 * period_multiplier,
    )

    ta_df["macd_line"] = macd.macd()
    ta_df["macd_signal"] = macd.macd_signal()
    ta_df["macd_hist"] = macd.macd_diff()

    rsi = RSIIndicator(df["close"], window=14 * period_multiplier)

    ta_df["rsi_14"] = rsi.rsi()

    return ta_df
