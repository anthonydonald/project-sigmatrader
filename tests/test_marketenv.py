import numpy as np
import pandas as pd
import pytest

from sigmatrader.marketenv.marketenv import Action, MarketEnv


@pytest.fixture
def market_env():
    closetimes = np.array(
        [
            "2023-12-27T23:59:59.999000000",
            "2023-12-28T23:59:59.999000000",
            "2023-12-29T23:59:59.999000000",
            "2023-12-30T23:59:59.999000000",
            "2023-12-31T23:59:59.999000000",
        ],
        dtype="datetime64[ns]",
    )
    opentimes = np.array(
        [
            "2023-12-27T00:00:00.000000000",
            "2023-12-28T00:00:00.000000000",
            "2023-12-29T00:00:00.000000000",
            "2023-12-30T00:00:00.000000000",
            "2023-12-31T00:00:00.000000000",
        ],
        dtype="datetime64[ns]",
    )
    close_prices = np.array([80, 100, 120, 140, 160])
    open_prices = np.array([75, 80, 100, 120, 140])
    mktdata_df = pd.DataFrame(
        {
            "closetime": closetimes,
            "opentime": opentimes,
            "close": close_prices,
            "open": open_prices,
        }
    )
    return MarketEnv(
        symbol="TESTSYM",
        df=mktdata_df,
        window_size=1,
        starting_cash=100,
        trading_fee=0,
        time_skip=0,
    )


def test_going_long(market_env):
    # GIVEN
    env = market_env
    action = Action.BUY
    env.reset()

    # WHEN
    env.step(action)
    # THEN
    assert env.get_price() == 100
    assert len(env.trades) == 1
    assert env.trades[-1]["direction"] == "BUY"
    assert env.trades[-1]["quantity"] == 100 / 80
    assert env.get_unrealised_pnl() == 25


def test_closing_long(market_env):
    # GIVEN
    env = market_env
    env.reset()

    # WHEN
    env.step(Action.BUY)
    env.step(Action.SELL)

    # THEN
    assert env.get_price() == 120
    assert len(env.trades) == 2
    assert env.trades[-1]["direction"] == "SELL"
    assert env.trades[-1]["quantity"] == 1.25
    assert env.realised_pnl == 25
    assert env.cash == 125
    assert env.get_unrealised_pnl() == 1.25 * -20
