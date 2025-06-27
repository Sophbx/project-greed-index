import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

TRANSACTION_COST = 0.0005  # 单边手续费 5bp
MAX_POSITION      = 1.0    # 最多 100% 多头
MIN_POSITION      = 0.0    # 允许全现金

# 文件：sp500_env.py
class SP500GreedEnv(gym.Env):
    def __init__(
        self,
        csv_path: str,
        lookback: int = 5,
        start_date: str | None = None,
        end_date: str | None = None,
    ):
        super().__init__()
        df = pd.read_csv(csv_path, parse_dates=["Date"]).sort_values("Date")

        # ---- ① 只保留 [start_date, end_date] ----
        if start_date:
            df = df[df["Date"] >= pd.Timestamp(start_date)]
        if end_date:
            df = df[df["Date"] <= pd.Timestamp(end_date)]
        df = df.reset_index(drop=True)

        self.df = df
        self.returns = df["close"].pct_change().fillna(0.0).values

        # 指标列可按需增删
        feats = [
            "norm_rsi",
            "norm_macd",
            "norm_zscore",
            "norm_vix",
            "greed_index",
        ]
        self.states_raw = (
            df[feats].fillna(method="ffill").astype("float32").values
        )

        self.lookback = lookback

        # 动作空间：0,1,2 三档仓位
        self.action_space = spaces.Discrete(3)

        # 观测空间：lookback 条状态平铺 + 当前仓位
        obs_dim = len(feats) * lookback + 1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.reset()

    def _get_obs(self):
        # 最近 lookback 天的特征拼接，再加当前仓位
        start = self.t - self.lookback + 1
        window = self.states_raw[start : self.t + 1].flatten()
        return np.concatenate([window, [self.position]]).astype(np.float32)

    def step(self, action: int):
        prev_position = self.position
        # 动作映射到目标仓位
        target = {0: 0.0, 1: 0.5, 2: 1.0}[int(action)]
        self.position = target

        # 手续费 = |Δ仓位| * cost
        cost = abs(self.position - prev_position) * TRANSACTION_COST

        # 当日收益
        daily_ret = self.returns[self.t + 1] * self.position
        reward = daily_ret - cost

        self.t += 1
        done = self.t >= len(self.df) - 2   # 留一天给 next_return
        obs = self._get_obs()
        info = {"position": self.position, "daily_return": daily_ret}

        return obs, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 从头开始，每次训练跑完整段时间序列
        self.t = self.lookback - 1
        self.position = 1.0   # 初始满仓，也可以设 0
        return self._get_obs(), {}

    def render(self):
        print(f"Day={self.t}  Pos={self.position:.2f}")
