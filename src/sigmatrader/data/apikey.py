import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())


@dataclass(frozen=True)
class APIKey:
    binance_api_key: str = os.getenv("binance_api_key")
    binance_pvt_key: str = os.getenv("binance_pvt_key")
    coinbase_api_key: str = os.getenv("coinbase_api_key")
    coinbase_pvt_key: str = os.getenv("coinbase_pvt_key")
    coinmarketcap_key: str = os.getenv("coinmarketcap_key")
