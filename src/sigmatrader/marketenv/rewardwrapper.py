from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from gymnasium import RewardWrapper

from sigmatrader.marketenv.metrics import TradeMetrics, delta_days


class TimeAdjRewardWrapper(RewardWrapper):
    def __init__(self, env, days_start, days_end):
        assert days_start < days_end
        self.days_start, self.days_end = days_start, days_end
        super().__init__(env)

    # Override the reward method
    def reward(self, reward):
        returns = 0
        if self.unwrapped.trades:
            last_trade = self.unwrapped.trades[-1]
            days_in_window = lambda days: self.days_start <= days <= self.days_end
            if (
                last_trade["status"] == "CLOSED"
                and last_trade["closed_timestamp"] == self.unwrapped.get_timestamp
            ):
                days = delta_days(
                    last_trade["closed_timestamp"] - last_trade["open_timestamp"]
                )
                if days_in_window(days) and last_trade["realised_pnl"] > 0:
                    returns += last_trade["realised_pnl"]
                returns += last_trade["realised_pnl"]
            else:
                days = delta_days(
                    self.unwrapped.get_timestamp() - last_trade["open_timestamp"]
                )
                if days_in_window(days) and self.unwrapped.get_unrealised_pnl() > 0:
                    returns += self.unwrapped.get_unrealised_pnl()
                returns += self.unwrapped.get_unrealised_pnl()

        return returns


class WeightedTradeTermReward2(RewardWrapper):
    def __init__(self, env, days_start, days_end, time_weight):
        assert days_start < days_end
        super().__init__(env)
        self.days_start, self.days_end = days_start, days_end
        self.time_weight = time_weight

    def reward(self, reward: float):
        trade_term_reward = 0
        if (trades := self.unwrapped.trades) and trades[-1]["status"] == "OPEN":
            days_open = delta_days(
                self.unwrapped.get_timestamp() - trades[-1].get("open_timestamp")
            )
            pnl_pct = self.unwrapped.get_unrealised_pnl() / (
                trades[-1]["price"] * trades[-1]["quantity"]
            )
            trade_term_reward += (
                ((self.days_start - abs(days_open - self.days_start)) / self.days_start)
                * self.time_weight
                / 2
            ) * pnl_pct
            trade_term_reward += (
                ((self.days_end - abs(days_open - self.days_end)) / self.days_end)
                * self.time_weight
                / 2
            ) * pnl_pct
        return reward * (1 - self.time_weight) + trade_term_reward
