from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache

import numpy as np
import pandas as pd
from binance.spot import Spot

from sigmatrader.data.apikey import APIKey

KLINES_COLUMNS = [
    "opentime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "closetime",
    "quotevolume",
    "numtrades",
    "takerbuyvolume",
    "takerbuyquotevolume",
    "ignore",
]

DEFAULT_DROP_COLUMNS = [
    "ignore",
    "quotevolume",
    "numtrades",
    "takerbuyvolume",
    "takerbuyquotevolume",
]

timestamp_to_datetime = np.vectorize(lambda t: datetime.fromtimestamp(t / 1000))
datetime_to_timestamp = lambda d: int(datetime.timestamp(d)) * 1000


@lru_cache
def fetch_data(pair, start_dt, end_dt, interval):

    period, term = int(interval[:-1]), interval[-1]
    delta = None
    match term:
        case "h":
            delta = timedelta(hours=period)
        case "d":
            delta = timedelta(days=period)
        case _:
            raise NotImplementedError("Only support hour/day interval.")

    client = Spot(api_key=APIKey.binance_api_key, api_secret=APIKey.binance_pvt_key)

    last_dt = start_dt
    data_frames = []
    while (last_dt + delta) < end_dt:

        klines_data = client.klines(
            symbol=pair,
            interval=interval,
            startTime=datetime_to_timestamp(last_dt),
            endTime=datetime_to_timestamp(end_dt),
            limit=1000,
        )
        klines_df = pd.DataFrame(klines_data, columns=KLINES_COLUMNS)

        klines_df["opentime"] = pd.to_datetime(
            timestamp_to_datetime(klines_df["opentime"])
        )
        klines_df["open"] = klines_df["open"].astype(float)
        klines_df["high"] = klines_df["high"].astype(float)

        klines_df["low"] = klines_df["low"].astype(float)
        klines_df["close"] = klines_df["close"].astype(float)
        klines_df["volume"] = klines_df["volume"].astype(float)
        klines_df["closetime"] = pd.to_datetime(
            timestamp_to_datetime(klines_df["closetime"])
        )
        klines_df["quotevolume"] = klines_df["quotevolume"].astype(float)
        klines_df["numtrades"] = klines_df["numtrades"].astype(int)
        klines_df["takerbuyvolume"] = klines_df["takerbuyvolume"].astype(float)
        klines_df["takerbuyquotevolume"] = klines_df["takerbuyquotevolume"].astype(
            float
        )
        klines_df["ignore"] = klines_df["ignore"].astype(object)

        data_frames.append(klines_df)

        last_dt = klines_df.iloc[-1, :]["closetime"]

    return (
        pd.concat(data_frames).reset_index(drop=True).drop(columns=DEFAULT_DROP_COLUMNS)
    )
