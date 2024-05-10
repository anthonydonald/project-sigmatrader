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

    model_cls = get_class(config.algo._target_)
    algo_name = model_cls.__name__

    run_path = f"runs/{runid}/{symbol}_{algo_name.lower()}"

    logging.basicConfig(
        filename=f"{run_path}/evaluate.log", level=logging.INFO, force=True
    )

    model_files = [
        config.get("modelfile") or f
        for f in os.listdir(f"runs/{runid}")
        if f.endswith(".zip") and f.startswith(f"{algo_name.lower()}_")
    ]
    logging.info(
        f"Evaluation for symbol: {symbol} with config: {config}, algo: {algo_name}, model_files: {model_files}"
    )

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
    transformer.fit(pd.concat([datasets["train"]]))
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
    for model_file in model_files:
        env = gym.make(env_id, **env_kwargs)
        # check_env(env)
        for wrap in config.wrappers:
            env = instantiate(wrap, env=env)

        env = make_vec_env(
            env_id=make_env,
            env_kwargs=env_kwargs,
            seed=np.random.randint(2**32 - 1, dtype="int64").item(),
            monitor_kwargs=dict(info_keywords=("date", "Trader return")),
        )
        if is_normalised := config.env.normalise:
            env = normalisation(env, run_path)

        # Load agent
        model_cls = get_class(config.algo._target_)
        algo_name = model_cls.__name__
        logging.info(f"Loading {algo_name} from {model_file}")
        name, ext = os.path.splitext(model_file)

        model = model_cls.load(
            f"runs/{runid}/{name}",
            env=env,
        )

        # Evaluate
        mean_reward, std_reward = evaluate_policy(
            model.policy,
            model.get_env(),
            config.n_eval_episodes,
        )
        evaluation_str = f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}"
        print(evaluation_str)
        logging.info(evaluation_str)

        vec_env = model.get_env()
        for ep in range(config.n_eval_episodes):
            obs = vec_env.reset()
            done = False
            while not done:
                action, _info = model.predict(obs, deterministic=config.deterministic)
                obs, reward, done, info = vec_env.step(action)
                logging.debug(f"reward: {reward}")
                if done:
                    metrics = info[-1]
                    print(f"Episode {ep} complete: {info[-1]}")
                    stats.append(metrics)
    stats_df = pd.DataFrame(stats)
    logging.info(stats_df)
    logging.info(stats_df.describe())
    # stats_df.describe().to_csv(f"{run_path}_stats_describe.csv")
    # stats_df.to_csv(f"{run_path}_stats.csv")


if __name__ == "__main__":
    main()
