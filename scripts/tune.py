from __future__ import annotations

from datetime import datetime
from typing import Any, Dict

import gymnasium as gym
import optuna
import pandas as pd
import torch
import torch as th
import torch.nn as nn
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from sigmatrader.dataloader import BinanceLoader
from sigmatrader.featurebuilder import MultiTransformer, TABuilder
from sigmatrader.marketenv.marketenv import MarketEnv
from sigmatrader.marketenv.rewardwrapper import (
    TimeAdjRewardWrapper,
    WeightedTradeTermReward,
    WeightedTradeTermReward2,
)
from sigmatrader.splitter import RelativeSplitter

symbol = "BTCUSDT"

config = {
    "loader": {
        "symbol": symbol,
        "start_date": datetime(2019, 9, 1),
        "end_date": datetime(2024, 3, 31),
        "interval": (input_interval := "1d"),
    }
}
env_kwargs = dict(
    symbol=symbol,
    window_size=10,
    starting_cash=1000,
    trading_fee=0.001,
    time_skip=0,
    target_min_term=14,
    render_mode="human",
)

loader = BinanceLoader(**config["loader"])
splitter = RelativeSplitter(train=0.8, validation=0.1, __test=0.1)

transformers = []
ta_features = TABuilder(input_interval)
transformers.append(ta_features)
transformer = MultiTransformer(transformers=transformers)

# Run data collection and processing
data = loader.load()

datasets = splitter.split(data)
train_data = transformer.fit_transform(datasets["train"])
validation_data = transformer.transform(datasets["validation"])

N_TRIALS = 50  # Maximum number of trials
N_JOBS = 1  # Number of jobs to run in parallel
N_STARTUP_TRIALS = 5  # Stop random sampling after N_STARTUP_TRIALS
N_EVALUATIONS = 2  # Number of evaluations during the training
N_TIMESTEPS = 100_000  # Training budget
# EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
EVAL_FREQ = 180
N_EVAL_ENVS = 6
N_EVAL_EPISODES = 5
TIMEOUT = int(60 * 120)  # 120 minutes

ENV_ID = "MarketEnv-v0"
# ENV_ID = "CartPole-v1"

ENV_KWARGS = dict(
    df=train_data,
    **env_kwargs,
)

EVAL_ENV_KWARGS = dict(
    df=validation_data,
    **env_kwargs,
)

DEFAULT_HYPERPARAMS = {
    "policy": "MlpPolicy",
}


def make_mkt_env(**kwargs) -> gym.Env:
    print(kwargs)
    return spec.make(**kwargs)


spec = gym.spec(ENV_ID)


def main():

    # Set pytorch num threads to 1 for faster training
    th.set_num_threads(1)
    # Select the sampler, can be random, TPESampler, CMAES, ...
    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=N_EVALUATIONS // 3
    )
    # Create the study and start the hyperparameter optimization
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    try:
        study.optimize(objective, n_trials=N_TRIALS, n_jobs=N_JOBS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("WeightedTradeTermReward2-a2c.csv")

    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)

    fig1.show()
    fig2.show()


# From rl-baselines3-zoo:
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/27e081eb24419ee843ae1c329b0482db823c9fc1/rl_zoo3/hyperparams_opt.py#L11
def sample_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for PPO hyperparams.

    :param trial:
    :return:
    """
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    # Uncomment for gSDE (continuous action)
    # sde_sample_freq = trial.suggest_categorical("sde_sample_freq", [-1, 8, 16, 32, 64, 128, 256])
    # Orthogonal initialization
    ortho_init = False
    # ortho_init = trial.suggest_categorical('ortho_init', [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])
    # lr_schedule = "constant"
    # Uncomment to enable learning rate schedule
    # lr_schedule = trial.suggest_categorical('lr_schedule', ['linear', 'constant'])
    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)

    # TODO: account when using multiple envs
    if batch_size > n_steps:
        batch_size = n_steps

    # Independent networks usually work best
    # when not working with images
    net_arch = {
        "tiny": dict(pi=[64], vf=[64]),
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=False,
        ),
    }


# From rl-baselines3-zoo:
# https://github.com/DLR-RM/rl-baselines3-zoo/blob/27e081eb24419ee843ae1c329b0482db823c9fc1/rl_zoo3/hyperparams_opt.py#L168
def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Sampler for A2C hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical(
        "gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999]
    )
    normalize_advantage = trial.suggest_categorical(
        "normalize_advantage", [False, True]
    )
    max_grad_norm = trial.suggest_categorical(
        "max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5]
    )
    # Toggle PyTorch RMS Prop (different from TF one, cf doc)
    use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
    gae_lambda = trial.suggest_categorical(
        "gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0]
    )
    n_steps = trial.suggest_categorical(
        "n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    )
    lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0, 1)
    # Uncomment for gSDE (continuous actions)
    # log_std_init = trial.suggest_float("log_std_init", -4, 1)
    ortho_init = trial.suggest_categorical("ortho_init", [False, True])
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
    # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
    # full_std = trial.suggest_categorical("full_std", [False, True])
    # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
    activation_fn_name = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

    # if lr_schedule == "linear":
    #     learning_rate = linear_schedule(learning_rate)  # type: ignore[assignment]

    net_arch = {
        "small": dict(pi=[64, 64], vf=[64, 64]),
        "medium": dict(pi=[256, 256], vf=[256, 256]),
    }[net_arch_type]

    # sde_net_arch = {
    #     None: None,
    #     "tiny": [64],
    #     "small": [64, 64],
    # }[sde_net_arch]

    activation_fn = {
        "tanh": nn.Tanh,
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "leaky_relu": nn.LeakyReLU,
    }[activation_fn_name]

    return {
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "normalize_advantage": normalize_advantage,
        "max_grad_norm": max_grad_norm,
        "use_rms_prop": use_rms_prop,
        "vf_coef": vf_coef,
        "policy_kwargs": dict(
            # log_std_init=log_std_init,
            net_arch=net_arch,
            # full_std=full_std,
            activation_fn=activation_fn,
            # sde_net_arch=sde_net_arch,
            ortho_init=ortho_init,
        ),
    }


# def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
#     """
#     Sampler for A2C hyperparameters.

#     :param trial: Optuna trial object
#     :return: The sampled hyperparameters for the given trial.
#     """
#     # Discount factor between 0.9 and 0.9999
#     gamma = 1.0 - trial.suggest_float("gamma", 0.9, 0.99, log=True)
#     max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
#     # 8, 16, 32, ... 1024
#     n_steps = 2 ** trial.suggest_int("exponent_n_steps", 3, 10)

#     ### YOUR CODE HERE
#     # TODO:
#     # - define the learning rate search space [1e-5, 1] (log) -> `suggest_float`
#     # - define the network architecture search space ["tiny", "small"] -> `suggest_categorical`
#     # - define the activation function search space ["tanh", "relu"]
#     learning_rate = trial.suggest_float("lr", 1e-5, 1, log=True)
#     net_arch = trial.suggest_categorical("net_arch", ["tiny", "small"])
#     activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

#     ### END OF YOUR CODE

#     # Display true values
#     trial.set_user_attr("gamma_", gamma)
#     trial.set_user_attr("n_steps", n_steps)

#     net_arch = [
#         (
#             {"pi": [64], "vf": [64]}
#             if net_arch == "tiny"
#             else {"pi": [64, 64], "vf": [64, 64]}
#         )
#     ]

#     activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU}[activation_fn]

#     return {
#         "n_steps": n_steps,
#         "gamma": gamma,
#         "learning_rate": learning_rate,
#         "max_grad_norm": max_grad_norm,
#         "policy_kwargs": {
#             "net_arch": net_arch,
#             "activation_fn": activation_fn,
#         },
#     }


class TrialEvalCallback(EvalCallback):
    """
    Callback used for evaluating and reporting a trial.

    :param eval_env: Evaluation environement
    :param trial: Optuna trial object
    :param n_eval_episodes: Number of evaluation episodes
    :param eval_freq:   Evaluate the agent every ``eval_freq`` call of the callback.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic policy.
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env,
        trial: optuna.Trial,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        deterministic: bool = True,
        verbose: int = 1,
    ):

        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
            verbose=verbose,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate policy (done in the parent class)
            super()._on_step()
            self.eval_idx += 1
            # Send report to Optuna
            print(self.last_mean_reward, self.eval_idx)
            self.trial.report(self.last_mean_reward, self.eval_idx)
            # Prune trial if need
            if self.trial.should_prune():
                self.is_pruned = True
                return False
        return True


def objective(trial: optuna.Trial) -> float:
    """
    Objective function using by Optuna to evaluate
    one configuration (i.e., one set of hyperparameters).

    Given a trial object, it will sample hyperparameters,
    evaluate it and report the result (mean episodic reward after training)

    :param trial: Optuna trial object
    :return: Mean episodic reward after training
    """

    kwargs = DEFAULT_HYPERPARAMS.copy()
    ### YOUR CODE HERE
    # TODO:
    # 1. Sample hyperparameters and update the default keyword arguments: `kwargs.update(other_params)`
    # 2. Create the evaluation envs
    # 3. Create the `TrialEvalCallback`

    # 1. Sample hyperparameters and update the keyword arguments
    kwargs.update(sample_a2c_params(trial))

    # kwargs.update(sample_ppo_params(trial))
    # Create the RL model
    reward_wrapper = WeightedTradeTermReward2
    reward_wrapper_kwargs = {"days_start": 14, "days_end": 44, "time_weight": 0.5}
    monitored_wrapper = lambda env: Monitor(
        reward_wrapper(env, **reward_wrapper_kwargs)
    )
    train_env = make_vec_env(
        make_mkt_env,
        N_EVAL_ENVS,
        env_kwargs=ENV_KWARGS,
        wrapper_class=monitored_wrapper,
    )
    model = A2C(env=train_env, **kwargs)

    # 2. Create envs used for evaluation using `make_vec_env`, `ENV_ID` and `N_EVAL_ENVS`
    eval_envs = make_vec_env(
        make_mkt_env,
        N_EVAL_ENVS,
        env_kwargs=EVAL_ENV_KWARGS,
        wrapper_class=monitored_wrapper,
    )

    # 3. Create the `TrialEvalCallback` callback defined above that will periodically evaluate
    # and report the performance using `N_EVAL_EPISODES` every `EVAL_FREQ`
    # TrialEvalCallback signature:
    # TrialEvalCallback(eval_env, trial, n_eval_episodes, eval_freq, deterministic, verbose)
    eval_callback = TrialEvalCallback(
        eval_envs, trial, N_EVAL_ENVS, EVAL_FREQ, deterministic=True, verbose=1
    )

    ### END OF YOUR CODE

    nan_encountered = False
    try:
        # Train the model
        model.learn(N_TIMESTEPS, callback=eval_callback)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True
    finally:
        # Free memory
        model.env.close()
        eval_envs.close()

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")

    if eval_callback.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_callback.last_mean_reward


if __name__ == "__main__":
    main()
