import pandas as pd
from sp500_env import SP500GreedEnv
from stable_baselines3 import PPO

env   = SP500GreedEnv("data/raw_data/Combined_data_2000.csv", lookback=5)
model = PPO.load("ppo_sp500_greed")

obs, _ = env.reset()
equity_curve = [1.0]   # 起始净值
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    equity_curve.append(equity_curve[-1] * (1 + info["daily_return"]))
    if done:
        break

# 保存每日净值
pd.DataFrame({"equity": equity_curve}).to_csv("rl_equity_curve.csv", index=False)
print("Final return:", equity_curve[-1] - 1)
