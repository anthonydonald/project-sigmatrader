import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from random import randrange
from typing import Any, Callable, Literal

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sigmatrader.marketenv.metrics import TradeMetrics, delta_days

logger = logging.getLogger(__name__)


class Action:
    HOLD = 0
    BUY = 1
    SELL = 2

    @staticmethod
    def name(action: int):
        name = None
        match action:
            case Action.HOLD:
                name = "HOLD"
            case Action.BUY:
                name = "BUY"
            case Action.SELL:
                name = "SELL"
            case _:
                raise ValueError(f"Unrecognised Action {action}")


class Position:
    NEUTRAL = "NEUTRAL"
    LONG = "LONG"
    SHORT = "SHORT"


RewardFunctorType = Callable[[IntEnum], float]


class MarketEnv(gym.Env):
    """Market environment with gym interface, inherited from Gymnasium's
    abstract class gymnasium.Env

    Action Space: Discrete(3)
    - 0: Hold
    - 1: Buy
    - 2: Sell
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        symbol: str,
        df: pd.DataFrame,
        window_size: int = 10,
        starting_cash: int = 1000,
        trading_fee: float = 0.001,
        loss_tolerance: float = 0.8,
        time_skip: int = 0,
        target_min_term: int = 0,
        render_mode=None,
    ):
        """_summary_

        Args:
            symbol (str): _description_
            df (pd.DataFrame): _description_
            window_size (int, optional): _description_. Defaults to 10.
            starting_cash (int, optional): _description_. Defaults to 1000.
            trading_fee (float, optional): _description_. Defaults to 0.
            loss_tolerance (float, optional): _description_. Defaults to 0.8.
            time_skip (int, optional): _description_. Defaults to 0.
            target_min_term (int, optional): _description_. Defaults to 0.
            render_mode (_type_, optional): _description_. Defaults to None.
        """
        self.symbol = symbol
        self.window_size = window_size
        self.target_min_term = target_min_term
        self.starting_cash = starting_cash
        self.trading_fee = trading_fee
        assert 0 <= loss_tolerance <= 1
        self.loss_tolerance = 1 - loss_tolerance

        self.time_skip = time_skip

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.set_market_data(df)

    def set_market_data(self, df: pd.DataFrame):
        self.df = df
        if set(["opentime", "closetime"]).issubset(df.columns):
            self.time_frame = df[["opentime", "closetime"]]
            self.df = df.drop(columns=["opentime", "closetime"])

        # self.action_space = gym.spaces.Discrete(len(Action))
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, len(self.df.columns)),
            dtype=np.float32,
        )

    def _get_obs(self):
        """Translates environment state into observation"""
        window_slice = slice(self._idx - self.window_size + 1, self._idx + 1)
        return self.df[window_slice].to_numpy(dtype=np.float32)

    def _get_info(self, is_final=False, **kwargs):
        """Auxiliary info getter"""
        full_slice = slice(self._start_idx, self._idx + 1)
        return {
            "date": self.get_timestamp(),
            "symbol": self.symbol,
            "Trader return": TradeMetrics.total_return.calc(
                self.book_df.iloc[full_slice]["value"]
            ),
            "Sharpe ratio": TradeMetrics.sharpe_ratio.calc(
                self.book_df.iloc[full_slice]
            ),
        } | kwargs  # TODO: finish/fix

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._start_idx = self.window_size - 1
        if self.time_skip:
            self._start_idx += randrange(self.time_skip)
            self.last_skip = max(self._start_idx - self.time_skip, 0)
        self._idx = self._start_idx
        self._last_trade_idx = None
        self.trades = []
        self.book_df = pd.DataFrame(
            self.time_frame.iloc[: self._start_idx + 1]["closetime"].values,
            columns=["timestamp"],
        )
        self.book_df["value"] = self.starting_cash
        self.book_df["unrealised_pnl"] = 0
        self.book_df["realised_pnl"] = 0

        self.realised_pnl = 0
        self.cash = self.starting_cash
        observation = self._get_obs()
        info = self._get_info()  # TODO: "auxiliary information"

        # TODO: render mode
        self._position = Position.NEUTRAL
        return observation, info

    def is_skip(self):
        return self.time_skip and self._idx % (self.time_skip + self.last_skip) == 0

    def in_trading_target_window(self):
        if self.target_min_term <= 0 or not self.trades:
            return True
        last_trade_dt = (
            self.trades[-1].get("closed_timestamp") or self.trades[-1]["open_timestamp"]
        )
        return (
            last_trade_dt + timedelta(days=self.target_min_term) >= self.get_timestamp()
        )

    def get_price(self, date=None):
        return self.df.iloc[self._idx]["close"]

    def get_timestamp(self):
        return self.time_frame.iloc[self._idx]["closetime"]

    def open_trade(self, direction):
        fee_adj = self.trading_fee if direction == "BUY" else -self.trading_fee
        fee_adj *= self.get_price()

        trade = {
            "direction": direction,
            "status": "OPEN",
            "open_timestamp": self.get_timestamp(),
            "price": (price := self.get_price() + fee_adj),
            "quantity": (quantity := (self.cash * 1) / price),
            "idx": self._idx,
            "closed_timestamp": None,
            "closed_price": None,
            "open_fee": fee_adj,
        }
        return trade

    def close_trade(self, trade):
        if (status := trade.get("status")) and status == "CLOSED":
            return
        fee_adj = -self.trading_fee if trade["direction"] == "BUY" else self.trading_fee
        fee_adj *= self.get_price()
        pnl = self.get_unrealised_pnl()
        trade["status"] = "CLOSED"
        trade["closed_timestamp"] = self.get_timestamp()
        trade["closed_price"] = self.get_price() + fee_adj
        trade["realised_pnl"] = pnl
        trade["closed_fee"] = fee_adj

        self.update_realised_pnl(pnl)

    def get_unrealised_pnl(self):
        """Calculate the profit or loss on open trade, adjusted for trading fees.

        Raises:
            RuntimeError: Current position direction not recognised.

        Returns:
            float: Unrealised profit (fee adjusted)
        """
        if not self.trades:
            return 0
        trade = self.trades[-1]
        if trade["status"] == "CLOSED":
            return 0
        entry_price = trade["price"]
        mkt_price = self.get_price()
        match (direction := trade["direction"]):
            case "BUY":
                pnl = (mkt_price - self.trading_fee * mkt_price) - entry_price
            case "SELL":
                pnl = entry_price - (mkt_price + self.trading_fee * mkt_price)
            case _:
                raise RuntimeError(f"Unexpected trade direction: {direction}")
        return pnl * trade["quantity"]

    def update_realised_pnl(self, pnl):
        self.cash += pnl
        self.realised_pnl += pnl

    def get_total_pnl(self):
        return self.get_unrealised_pnl() + self.realised_pnl

    def take_action(self, action):
        if self.is_skip():
            logging.debug("Skipping take_action at: {self._idx}")
            self.last_skip = self._idx
            return
        if self.in_trading_target_window():

            match (action, self._position):
                case (
                    (Action.HOLD, _)
                    | (Action.BUY, Position.LONG)
                    | (Action.SELL, Position.SHORT)
                ):
                    pass
                case (Action.BUY, position):
                    if position == Position.SHORT and self.trades:
                        # TODO: close
                        self.close_trade(self.trades[-1])
                    self.trades.append(self.open_trade("BUY"))
                    self._position = Position.LONG
                case (Action.SELL, position):
                    if position == Position.LONG and self.trades:
                        self.close_trade(self.trades[-1])
                    self.trades.append(self.open_trade("SELL"))
                    self._position = Position.SHORT
                case _:
                    raise RuntimeError(f"Unrecognised action {action}.")

    def step(self, action):
        assert self.action_space.contains(action)

        terminated = truncated = False
        terminated = self._idx >= len(self.df) - 1

        if not terminated:
            self.take_action(action=action)
            self._idx += 1
        elif self.trades and (last_trade := self.trades[-1])["status"] == "OPEN":
            self.close_trade(last_trade)

        terminated = self.cash < 0 or terminated

        self.update_book_metrics()
        if not terminated:
            terminated = self.book_df.iloc[self._idx, :]["value"] < (
                self.starting_cash * self.loss_tolerance
            )
        reward = self.reward(action)
        metrics_map = {}
        if (
            # self.render_mode == "human" and
            terminated
            or truncated
        ):
            metrics_map = self.log_metrics()

        info = self._get_info(action=action, reward=reward, **metrics_map)
        observation = self._get_obs()
        return observation, reward, terminated, truncated, info

    def reward(self, action: int) -> float:
        returns = 0
        open_px = self.df.iloc[self._idx]["open"]
        close_px = self.df.iloc[self._idx]["close"]
        match action:
            case Action.BUY:
                returns = (close_px - open_px) / open_px
            case Action.SELL:
                returns = -(close_px - open_px) / open_px
        return returns

    def update_book_metrics(self):
        self.book_df.loc[len(self.book_df)] = {
            "timestamp": self.get_timestamp(),
            "value": self.cash + self.get_unrealised_pnl(),
            "unrealised_pnl": self.get_unrealised_pnl(),
            "realised_pnl": self.cash - self.starting_cash,
        }

    def log_metrics(self):
        metrics_map = {}
        metrics_print_out = []
        full_slice = slice(self._start_idx, self._idx + 1)
        mkt_data_df = pd.DataFrame(
            zip(
                self.df.iloc[full_slice]["close"].values,
                self.time_frame.iloc[full_slice]["closetime"].values,
            ),
            columns=["value", "timestamp"],
        )
        metrics_print_out.append(
            TradeMetrics.total_return.report(
                "Market Return: ", self.df.iloc[full_slice]["close"]
            )
        )
        metrics_map["mkt_return"] = TradeMetrics.total_return.calc(
            self.df.iloc[full_slice]["close"]
        )
        metrics_print_out.append(
            TradeMetrics.mkt_sharpe_ratio.report("Market Sharpe ratio: ", mkt_data_df)
        )
        metrics_map["mkt_sharpe_ratio"] = TradeMetrics.mkt_sharpe_ratio.calc(
            mkt_data_df
        )
        # metrics = [
        #     TradeMetrics.total_return.report(
        #         "Market Return: ", self.df.iloc[full_slice]["close"]
        #     ),
        #     TradeMetrics.mkt_sharpe_ratio.report("Market Sharpe ratio: ", mkt_data_df),
        # ]
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df["closed_timestamp"] = trades_df["closed_timestamp"].fillna(
                self.get_timestamp()  # if trade still open, consider current date for metrics
            )
            metrics_print_out.extend(
                [
                    TradeMetrics.total_return.report(
                        "Trader return: ", self.book_df.iloc[full_slice]["value"]
                    ),
                    TradeMetrics.sharpe_ratio.report(
                        "Sharpe ratio: ", self.book_df.iloc[full_slice]
                    ),
                    TradeMetrics.holding_time_mean_days.report(
                        "Holding time mean days: ", trades_df, fmt="5.2f"
                    ),
                    TradeMetrics.number_of_trades.report(
                        "Number of trades: ", trades_df
                    ),
                ]
            )
            metrics_map["trader_return"] = TradeMetrics.total_return.calc(
                self.book_df.iloc[full_slice]["value"]
            )
            metrics_map["trader_sharpe_ratio"] = TradeMetrics.sharpe_ratio.calc(
                self.book_df.iloc[full_slice]
            )
            metrics_map["holding_time_mean_days"] = (
                TradeMetrics.holding_time_mean_days.calc(trades_df)
            )
            metrics_map["num_trades"] = TradeMetrics.number_of_trades.calc(mkt_data_df)
        else:
            metrics_print_out.append("No trades")
        metrics_print_out.append("\n")
        for m in metrics_print_out:
            logger.info(m)
            print(m)
        return metrics_map

    def metrics_map(self):
        metrics_map = {}
        full_slice = slice(self._start_idx, self._idx + 1)
        mkt_data_df = pd.DataFrame(
            zip(
                self.df.iloc[full_slice]["close"].values,
                self.time_frame.iloc[full_slice]["closetime"].values,
            ),
            columns=["value", "timestamp"],
        )
        metrics_map["Market Return"] = (
            TradeMetrics.total_return.calc(self.df.iloc[full_slice]["close"]),
        )
        metrics = [
            TradeMetrics.total_return.report(
                "Market Return: ", self.df.iloc[full_slice]["close"]
            ),
            TradeMetrics.mkt_sharpe_ratio.report("Market Sharpe ratio: ", mkt_data_df),
        ]
        if self.trades:
            trades_df = pd.DataFrame(self.trades)
            trades_df["closed_timestamp"] = trades_df["closed_timestamp"].fillna(
                self.get_timestamp()  # if trade still open, consider current date for metrics
            )
            metrics.extend(
                [
                    TradeMetrics.total_return.report(
                        "Trader return: ", self.book_df.iloc[full_slice]["value"]
                    ),
                    TradeMetrics.sharpe_ratio.report(
                        "Sharpe ratio: ", self.book_df.iloc[full_slice]
                    ),
                    TradeMetrics.holding_time_mean_days.report(
                        "Holding time mean days: ", trades_df, fmt="5.2f"
                    ),
                    TradeMetrics.number_of_trades.report(
                        "Number of trades: ", trades_df
                    ),
                ]
            )
        else:
            metrics.append("No trades")
        metrics.append("\n")
        for m in metrics:
            logger.info(m)
            print(m)

    def render(self):
        if self.render_mode == "human":
            self.log_metrics()

    def render_all(self):
        full_slice = slice(self._start_idx, self._idx + 1)
        df = self.book_df.iloc[full_slice]
        df["price"] = self.df["close"]
        df["cash"] = df["value"] - df["unrealised_pnl"]
        df = df.set_index("timestamp")

        fig, axes = plt.subplots(figsize=(18, 14), nrows=3)
        df.plot(y="price", use_index=True, ax=axes[0], secondary_y=True, color="black")
        df.plot(
            y="value",
            use_index=True,
            ax=axes[1],
            style="--",
            color="lightgrey",
        )
        df.plot(
            y="cash",
            use_index=True,
            ax=axes[1],
            color="black",
        )
        for trade in self.trades:
            idx = trade["open_timestamp"]
            match trade:
                case {"direction": "BUY"}:
                    plt.plot(idx, trade["price"], "g^")
                    # plt.text(
                    #     idx,
                    #     trade["price"] + 0.1,
                    #     trade["quantity"],
                    #     c="green",
                    #     fontsize=8,
                    #     horizontalalignment="center",
                    #     verticalalignment="center",
                    # )
                    if trade["status"] == "closed":
                        plt.plot(trade["closed_timestamp"], trade["closed_price"], "rv")
                        # plt.text(
                        #     trade["closed_timestamp"],
                        #     trade["closed_price"] + 0.1,
                        #     trade["quantity"],
                        #     c="red",
                        #     fontsize=8,
                        #     horizontalalignment="center",
                        #     verticalalignment="center",
                        # )
                case {"direction": "SELL"}:
                    plt.plot(idx, trade["price"], "rv")
                    # plt.text(
                    #     idx,
                    #     trade["price"] + 0.1,
                    #     trade["quantity"],
                    #     c="red",
                    #     fontsize=8,
                    #     horizontalalignment="center",
                    #     verticalalignment="center",
                    # )
                    if trade["status"] == "closed":
                        plt.plot(trade["closed_timestamp"], trade["closed_price"], "g^")
                        # plt.text(
                        #     trade["closed_timestamp"],
                        #     trade["closed_price"] + 0.1,
                        #     trade["quantity"],
                        #     c="green",
                        #     fontsize=8,
                        #     horizontalalignment="center",
                        #     verticalalignment="center",
                        # )
                case _:
                    pass

        # return (
        #     df["market_value"].pct_change().dropna(),
        #     df["price"].pct_change().dropna(),
        # )
