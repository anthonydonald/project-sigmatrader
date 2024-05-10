import os

from stable_baselines3.common.vec_env import VecEnv, VecNormalize


def setup_dir(dir: str):
    if not os.path.exists(dir):
        os.makedirs(dir)

    return dir


def normalisation(env: VecEnv, log_dir: str | None = None) -> VecEnv:
    # Path to pre-trained model normalisation stats
    if log_dir:
        norm_stats_path = os.path.join(log_dir, "vec_normalise.pkl")
        print(norm_stats_path)
        if os.path.exists(norm_stats_path):
            print(f"Loading stats from:{norm_stats_path}")
            env = VecNormalize.load(norm_stats_path, env)
            env.training = False
            env.norm_reward = False
    else:
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
    return env
