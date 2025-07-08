#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal-based position-sizing back-test  - 完整版
------------------------------------------------
● 信号含义
    0 = 无信号        → 维持前一日权重
    1 = signal_risk   → 0.0  （空仓）
    2 = signal_buy    → 1.5  （加仓）
    3 = both_signal   → 1.0  （折衷）
● 交易规则
    - 信号当日生成目标权重，T+1 调仓生效
    - 本示例不含手续费；可用 FEE_BPS 调整
● 功能扩展
    - 统计权重变动天数 (= 交易次数)
    - 可设定若干 PERIODS 做分段回测
    - 打印每一次调仓区间的策略收益 (Trade-by-Trade Returns)
------------------------------------------------
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# ========= 用户可调参数 ========= #
CSV_FILE      = Path("data/raw_data/greed_index_ml.csv")
FEE_BPS       = 0.0          # 双边手续费基点；2 bp = 0.0002
START_DATE    = "2005-01-03"
TICKER        = "SPY"
USE_ADJ_CLOSE = True

# ★★ 若要分段评估，填写 (start, end) 列表；留空 = 仅跑全样本
PERIODS = [
    ("2005-01-03", "2010-12-31"),
    ("2011-01-01", "2015-12-31"),
    ("2016-01-01", "2020-12-31"),
    ("2021-01-01", "2025-07-07"),
]
# ================================= #

def load_data(csv_path: Path, start_date: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.loc[df["Date"] >= start_date].copy()
    df.set_index("Date", inplace=True)
    needed_cols = {"close", "signal"}
    if not needed_cols.issubset(df.columns):
        raise KeyError(f"缺少列：{needed_cols - set(df.columns)}")
    return df

def fetch_adj_close(idx, start, end):
    data = yf.download(idx, start=start, end=end, auto_adjust=True, progress=False)
    col  = "Adj Close" if "Adj Close" in data.columns else "Close"
    ser  = data[[col]].squeeze("columns")
    ser.name = "adj_close"
    return ser

# ---------- 逐笔交易日志 ---------- #
def build_trade_log(nav_strategy: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    """根据权重变动日，生成逐笔收益日志"""
    trade_idx = turnover[turnover > 0].index
    if trade_idx.empty:
        return pd.DataFrame(columns=["Trade_Return"])

    prev_nav = nav_strategy.iloc[0]
    rows = []
    for dt in trade_idx:
        ret = nav_strategy.loc[dt] / prev_nav - 1
        rows.append({"Date": dt, "Trade_Return": ret})
        prev_nav = nav_strategy.loc[dt]

    return pd.DataFrame(rows).set_index("Date")

# ---------- 核心计算 ---------- #
def calc_nav(df: pd.DataFrame, fee_bps: float, use_adj: bool = True):
    # 价格与日收益
    if use_adj:
        adj = fetch_adj_close(
            TICKER,
            df.index[0].strftime("%Y-%m-%d"),
            (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        )
        price = adj.reindex(df.index).ffill()
        if price.isna().any():
            raise ValueError("Adj Close 有缺口，无法对齐日期")
    else:
        price = df["close"]

    daily_ret = price.pct_change().fillna(0.0)

    # 信号 → 当日目标权重
    mapped = np.select(
        [df["signal"] == 1, df["signal"] == 2, df["signal"] == 3],
        [0.0, 1.0, 0.8],           # risk / buy / both
        default=1.0
    )
    weight_today = pd.Series(mapped, index=df.index)
    weight_today.iloc[0] = 1.0                 # 初始满仓
    weight_today.ffill(inplace=True)

    # T+1 生效
    trade_weight = weight_today.shift(1).fillna(1.0)

    # 成本 & 交易次数
    turnover    = trade_weight.diff().abs().fillna(0.0)
    cost        = turnover * (fee_bps / 10000.0)
    trade_days  = int((turnover > 0).sum())

    # 策略 / 基准收益
    strat_ret   = trade_weight * daily_ret - cost
    buyhold_ret = daily_ret

    nav = pd.DataFrame({
        "Strategy": (1 + strat_ret).cumprod(),
        "Buy&Hold": (1 + buyhold_ret).cumprod()
    })
    nav.index.name = "Date"

    # 逐笔交易日志
    trade_log = build_trade_log(nav["Strategy"], turnover)

    return nav, strat_ret, buyhold_ret, trade_days, trade_log

# ---------- 绩效指标 ---------- #
def perf_summary(nav: pd.Series, ret: pd.Series, annual_factor: int = 252):
    total_return = nav.iloc[-1] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = ret.std() * np.sqrt(annual_factor)
    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    return total_return, annual_return, annual_vol, sharpe, max_dd

# ---------- 打包回测 ---------- #
def backtest(df_slice: pd.DataFrame, label: str = ""):
    nav, strat_ret, bh_ret, trade_days, trade_log = calc_nav(
        df_slice, FEE_BPS, USE_ADJ_CLOSE)

    # 绩效打印
    metrics = pd.DataFrame(
        {
            "Metric": ["Total Return", "Annualized Return", "Annualized Vol",
                       "Sharpe", "Max Drawdown"],
            "Strategy": perf_summary(nav["Strategy"], strat_ret),
            "Buy & Hold": perf_summary(nav["Buy&Hold"], bh_ret)
        }
    )
    print(f"\n=== Performance Summary {label} ===")
    print(metrics.to_string(index=False,
          float_format=lambda x: f"{x:>8.2%}" if abs(x) < 1 else f"{x:>8.3f}"))
    print(f"Trades executed (weight change days): {trade_days}")

    # 逐笔交易收益打印
    if not trade_log.empty:
        print("\n-- Trade-by-Trade Returns --")
        print(trade_log.to_string(float_format=lambda x: f"{x:.2%}"))
    else:
        print("\n-- Trade-by-Trade Returns --\n无交易发生")

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 4))
    nav.plot(ax=ax, title=f"Equity Curve {label}")
    ax.set_ylabel("NAV (Start = 1.0)")
    plt.tight_layout()
    plt.show()

# ---------- 主入口 ---------- #
def main():
    df = load_data(CSV_FILE, START_DATE)

    # ① 全样本
    backtest(df, f"(Full Sample) {df.index[0].date()} – {df.index[-1].date()}")

    # ② 分段
    for (s, e) in PERIODS:
        mask = (df.index >= s) & (df.index <= e)
        if mask.any():
            backtest(df.loc[mask], f"({s} ~ {e})")
        else:
            print(f"\n[Warning] 区间 {s} ~ {e} 无数据")

if __name__ == "__main__":
    main()
