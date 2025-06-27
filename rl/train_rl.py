import os
import numpy as np
from stable_baselines3 import PPO
from sp500_env import SP500GreedEnv

CSV_PATH = "data/Combined_data_2000.csv"

env = SP500GreedEnv(CSV_PATH, lookback=5)

# 让 SB3 自己包一层 VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
vec_env = DummyVecEnv([lambda: env])

model = PPO("MlpPolicy",
            vec_env,
            verbose=1,
            tensorboard_log="./tensorboard",
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.0)

model.learn(total_timesteps=200_000)
model.save("ppo_sp500_greed")
