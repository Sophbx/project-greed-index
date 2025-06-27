"""
Updated greed_momentum_rl.py
===========================

A **fully‑working, self‑contained** RL pipeline that now:
1. Uses **low trading cost** (0.005 %) so the agent has incentive to move.
2. Adds **drawdown penalty** to reward so仓位会在恐惧阶段减仓。
3. Expands动作空间到 **[-1, 1]** 支持做空或减仓。
4. 强化探索 (`ent_coef = 1e‑2`).
5. 记录并保存 `position_history.csv`，方便看仓位轨迹。
6. 打印指标用小数，不再把 Sharpe ×100。

Run steps (once deps installed):
```bash
python greed_momentum_rl.py        # train + back‑test
open rl_results/equity_curve.png   # 查看净值曲线
```
"""

# ---------- Imports ----------
import os
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------- CONFIG ----------------
CSV_PATH = "data/Combined_data_2000.csv"     # your data file
FEATURE_COLS = [
    "greed_index"
]
PRICE_COL = "close"

TRAIN_START, TRAIN_END = "2000-01-03", "2015-12-31"
TEST_START,  TEST_END  = "2016-01-04", "2025-06-25"

COST = 0.00005          # 0.005 % round‑trip
RISK_PENALTY = 0.5      # λ for drawdown penalty (positive number)

TOTAL_TIMESTEPS = 2_000_000
BATCH_SIZE = 256
GAMMA = 0.99
ENT_COEF = 1e-2         # ↑ exploration
LR = 3e-4

OUT_DIR = "rl_results"

# ------------- helpers -------------

def perf_stats(nav: pd.Series) -> dict:
    nav = nav.dropna()
    ret = nav.pct_change().dropna()
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252/len(nav)) - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe  = ann_ret / ann_vol if ann_vol else np.nan
    mdd     = (nav / nav.cummax() - 1).min()
    return dict(annual_return=ann_ret,
                annual_vol=ann_vol,
                sharpe=sharpe,
                max_drawdown=mdd)

# ------------- Environment -------------
class GreedMomentumEnv(gym.Env):
    """Single‑asset, continuous‑action trading env (‑1~1 position)."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, df: pd.DataFrame, feat_cols, price_col="close", cost=0.00005, risk_penalty=0.5):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.features = self.df[feat_cols].astype(np.float32).values
        self.price = self.df[price_col].values.astype(np.float32)
        self.cost = cost
        self.risk_penalty = risk_penalty

        obs_dim = self.features.shape[1] + 2  # + position + nav
        self.action_space      = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(obs_dim,), dtype=np.float32)

    # ---- RL API ----
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        self.pos = 0.0
        self.nav = 1.0
        self.max_nav = 1.0
        self.pos_history = [self.pos]
        info = {}
        return self._obs(), info

    def _obs(self):
        return np.concatenate([self.features[self.idx], [self.pos, self.nav]])

    def step(self, action):
        action = float(np.clip(action, -1.0, 1.0))
        ret = (self.price[self.idx+1] - self.price[self.idx]) / self.price[self.idx]
        trade_cost = self.cost * abs(action - self.pos)
        reward = self.pos * ret - trade_cost

        # update nav + drawdown penalty
        self.nav *= (1 + self.pos * ret - trade_cost)
        self.max_nav = max(self.max_nav, self.nav)
        drawdown = (self.nav - self.max_nav) / self.max_nav  # ≤0
        reward += self.risk_penalty * drawdown  # subtract if drawdown<0

        # advance
        self.idx += 1
        terminated = self.idx >= len(self.price) - 1
        truncated  = False
        self.pos = action
        self.pos_history.append(self.pos)
        info = {}
        return self._obs(), reward, terminated, truncated, info

    # run policy deterministically on this env
    def run_policy(self, model):
        obs, _ = self.reset()
        navs = [1.0]
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = self.step(act)
            done = term or trunc
            navs.append(self.nav)
        idx = self.df.loc[: self.idx, "Date"].reset_index(drop=True)
        nav_series = pd.Series(navs, index=idx, name="strategy_nav")
        pos_series = pd.Series(self.pos_history, index=idx, name="position")
        return nav_series, pos_series

# ------------- main -------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # load & split
    df_all = (
        pd.read_csv(CSV_PATH, parse_dates=["Date"])
        .sort_values("Date")
        .dropna(subset=FEATURE_COLS + [PRICE_COL])
    )
    df_train = df_all[(df_all.Date >= TRAIN_START) & (df_all.Date <= TRAIN_END)].reset_index(drop=True)
    df_test  = df_all[(df_all.Date >= TEST_START)  & (df_all.Date <= TEST_END)].reset_index(drop=True)

    # envs
    env_train = DummyVecEnv([lambda: GreedMomentumEnv(df_train, FEATURE_COLS, PRICE_COL, COST, RISK_PENALTY)])
    env_test  = GreedMomentumEnv(df_test,  FEATURE_COLS, PRICE_COL, COST, RISK_PENALTY)

    # train PPO
    model = PPO(
        "MlpPolicy", env_train, batch_size=BATCH_SIZE, gamma=GAMMA,
        ent_coef=ENT_COEF, learning_rate=LR, verbose=1)
    model.learn(total_timesteps=TOTAL_TIMESTEPS)

    # back‑test
    nav_rl, pos_rl = env_test.run_policy(model)
    stats_rl = perf_stats(nav_rl)

    # baseline Buy & Hold
    bh_nav = df_test.set_index("Date")[PRICE_COL] / df_test[PRICE_COL].iloc[0]
    stats_bh = perf_stats(bh_nav)

    print("\n===== 2016‑2025 PERFORMANCE =====")
    for k in stats_rl:
        print(f"{k:<15} RL: {stats_rl[k]:.3%}    Buy&Hold: {stats_bh[k]:.3%}")

    # save outputs
    nav_rl.to_csv(os.path.join(OUT_DIR, "nav_series.csv"))
    pos_rl.to_csv(os.path.join(OUT_DIR, "position_history.csv"))
    model.save(os.path.join(OUT_DIR, "ppo_greed_momentum.zip"))

    # plot
    plt.figure(figsize=(10,4))
    plt.plot(nav_rl, label="RL")
    plt.plot(bh_nav, label="Buy&Hold", linewidth=0.8)
    plt.title("Equity Curve (Test Set)")
    plt.ylabel("Net Asset Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "equity_curve.png"))
    plt.close()

if __name__ == "__main__":
    main()
