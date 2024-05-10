import logging
import os
from datetime import datetime

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from utils import normalisation

import sigmatrader.marketenv
from sigmatrader.dataloader import BinanceLoader
from sigmatrader.featurebuilder import MultiTransformer, TABuilder
from sigmatrader.marketenv.metrics import TradeMetrics
from sigmatrader.splitter import RelativeSplitter

OmegaConf.register_new_resolver("eval", lambda x: eval(x))


@hydra.main(config_path="config", config_name="evaluate", version_base="1.3")
def main(config: DictConfig):
    symbol = config.symbol
    runid = config.runid

    # Data collection and processing
    loader = BinanceLoader(**config.loader)
    splitter = RelativeSplitter(**config.splitter)

    transformers = []
    ta_features = TABuilder()
    transformers.append(ta_features)

    transformer = MultiTransformer(transformers=transformers)

    # Execute pipeline
    data = loader.load()
    datasets = splitter.split(data)
    transformer.fit(pd.concat([datasets["train"], datasets["validation"]]))
    test_data = transformer.transform(datasets["test"])

    # Create RL environment
    env_id = config.env.env_id
    spec = gym.spec(env_id)

    def make_env(**kwargs) -> gym.Env:
        print(kwargs)
        return spec.make(**kwargs)

    env_kwargs = {
        "symbol": symbol,
        "df": test_data,
        "render_mode": "human",
        **config.env.kwargs,
    }
    stats = []
    env = gym.make(env_id, **env_kwargs)
    for wrap in config.wrappers:
        env = instantiate(wrap, env=env)

    env = make_vec_env(
        env_id=make_env,
        env_kwargs=env_kwargs,
        seed=np.random.randint(2**32 - 1, dtype="int64").item(),
        monitor_kwargs=dict(info_keywords=("date", "Trader return")),
    )

    for ep in range(1):
        obs = env.reset()
        done = False
        # step_i = 0
        while not done:
            action = env.action_space.sample()
            # action = 1 if step_i ==0 else 0
            obs, reward, done, info = env.step([1])
            logging.debug(f"reward: {reward}")
            if done:
                metrics = info[-1]
                print(f"Episode {ep} complete: {info[-1]}")
                stats.append(metrics)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(f"market_{symbol}_stats.csv")

    stats_df.describe().to_csv(f"market_{symbol}_stats_describe.csv")


if __name__ == "__main__":
    main()
