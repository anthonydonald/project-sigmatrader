from gymnasium.envs.registration import register

# from sigmatrader.marketenv.cryptoenv import CryptoEnv
from sigmatrader.marketenv.marketenv import MarketEnv

register(
    id="MarketEnv-v0",
    entry_point="sigmatrader.marketenv.marketenv:MarketEnv",
)
