import logging
import os
from datetime import datetime

import gymnasium as gym
import hydra
import numpy as np
import pandas as pd
import torch.nn as nn
import wandb
from hydra.utils import get_class, instantiate
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import VecEnv, VecNormalize
from stable_baselines3.ppo import PPO
from utils import normalisation, setup_dir
from wandb.integration.sb3 import WandbCallback

import sigmatrader.marketenv
from sigmatrader.dataloader import BinanceLoader
from sigmatrader.featurebuilder import MultiTransformer, TABuilder
from sigmatrader.marketenv.rewardwrapper import TimeAdjRewardWrapper
from sigmatrader.splitter import RelativeSplitter

OmegaConf.register_new_resolver("eval", lambda x: eval(x))


@hydra.main(config_path="config", config_name="train", version_base="1.3")
def main(config: DictConfig):
    symbol = config.symbol
    algo_name = (model_cls := get_class(config.algo._target_)).__name__
    seeds = config.get("seeds", [np.random.randint(2**32 - 1, dtype="int64").item()])
    mean_eval_rewards = []

    runid = config.runid

    for seed in seeds:
        logging.info(f"Running with seed: {seed}")
        model_name = f"{algo_name.lower()}_{symbol}_{seed}"
        try:
            if wandb_kwargs := config.get("wandb"):
                run = wandb.init(name=model_name, group=runid, **wandb_kwargs)
                logging.debug(
                    f"Monitoring with wandb, group set to {runid}, wandb runid: {run.id}"
                )

            run_path = f"runs/{runid}/{symbol}_{algo_name.lower()}"
            setup_dir(run_path)
            logging.basicConfig(
                filename=f"{run_path}/train.log", level=logging.INFO, force=True
            )
            logging.info(f"Training config: {config}")

            # Data collection and processing setup
            loader = BinanceLoader(**config.loader)
            splitter = RelativeSplitter(**config.splitter)

            transformers = []
            ta_features = TABuilder(config.loader.interval)
            transformers.append(ta_features)
            transformer = MultiTransformer(transformers=transformers)

            # Run data collection and processing
            logging.info(f"Loading and processing data for {symbol}")
            data = loader.load()

            datasets = splitter.split(data)
            train_data = transformer.fit_transform(pd.concat([datasets["train"]]))
            validation_data = transformer.transform(datasets["validation"])

            # Create RL environment
            env_id = config.env.env_id
            spec = gym.spec(env_id)

            def make_env(**kwargs) -> gym.Env:
                print(kwargs)
                return spec.make(**kwargs)

            env_kwargs = {"symbol": symbol, "df": train_data, **config.env.kwargs}

            set_random_seed(seed=seed)

            eval_env_kwargs = {
                "symbol": symbol,
                "df": validation_data,
                "render_mode": "human",
                **config.env.kwargs,
            }

            monitored_wrapper = None
            if reward_wrapper_cfg := config.get("reward_wrapper"):
                reward_wrapper = get_class(reward_wrapper_cfg._target_)
                monitored_wrapper = lambda env: Monitor(
                    reward_wrapper(env, **reward_wrapper_cfg.kwargs)
                )

            env = make_vec_env(
                make_env,
                seed=seed,
                env_kwargs=env_kwargs,
                wrapper_class=monitored_wrapper,
                n_envs=config.get("n_envs", 2),
            )
            eval_env = make_vec_env(
                env_id=make_env,
                env_kwargs=eval_env_kwargs,
                wrapper_class=monitored_wrapper,
            )

            # Normalise?
            if is_normalised := config.env.normalise:
                env = normalisation(env)

            if policy_kwargs := config.algo.get("policy_kwargs"):
                policy_kwargs = eval(policy_kwargs)

            # Instantiate agent
            model = instantiate(
                config.algo,
                env=env,
                tensorboard_log=(tb_log_path := f"runs/{runid}"),
                verbose=1,
                policy_kwargs=policy_kwargs,
            )

            logging.info(f"Instantiated model: {algo_name}")
            cbs = []
            if is_model_saved := config.model.save:
                checkpoint_cb = CheckpointCallback(
                    save_freq=config.total_timesteps,
                    save_path=run_path,
                    name_prefix=f"checkpoint",
                    save_vecnormalize=is_normalised,
                )
                cbs.append(checkpoint_cb)
            if wandb_kwargs:
                wandb_cb = WandbCallback(
                    gradient_save_freq=100, model_save_path=f"models/{runid}", verbose=2
                )
                cbs.append(wandb_cb)

            # Train agent
            if not (skip_learning := config.skip_learn):

                logging.info(
                    f"Start training agent. See tensorboard logs: `tensorboard --log-dir {tb_log_path}`"
                )
                model.learn(
                    total_timesteps=config.total_timesteps,
                    callback=cbs,
                    progress_bar=config.progress_bar,
                )

            # Save agent
            if config.model.save:
                tag = "untrained" if skip_learning else seed
                model_dir = f"runs/{runid}/{model_name}"
                logging.info(f"Saving model to {model_dir}")
                model.save(model_dir)

                if is_normalised:
                    vec_normalise = model.get_vec_normalize_env()
                    vec_normalise.save(os.path.join(run_path, "vec_normalise.pkl"))
            env.close()

            # Evaluation
            # Normalise?
            if is_normalised:
                eval_env = normalisation(eval_env, run_path)

            eval_model = model_cls.load(f"{model_dir}.zip", env=eval_env)

            mean_reward, std_reward = evaluate_policy(
                model=eval_model,
                env=eval_env,
                n_eval_episodes=config.eval.n_eval_episodes,
            )
            eval_env.close()

            logging.info(
                f"Evaluation mean_reward: {mean_reward}, std_reward: {std_reward}"
            )
            mean_eval_rewards.append(mean_reward)
            if wandb_kwargs:
                run.finish()
        except KeyboardInterrupt:
            logging.info("Keyboard instructed stop.")
    print(
        median_eval_str := f"The median eval reward across evals {np.median(mean_eval_rewards)}"
    )
    logging.info(print(median_eval_str))


if __name__ == "__main__":
    main()
