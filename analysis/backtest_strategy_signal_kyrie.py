#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal-based position-sizing back-test  -- REFACTORED
-----------------------------------------------------
改动摘要：
1. 仅下载一次 Yahoo 复权价，避免每个分段重复请求
2. 用价格日历做 inner-join，**剔除周末 / 停牌日**，杜绝缺口前向填充
3. 费用按“双边 bps”计费，可选滑点
4. Sharpe = 年化日均超额收益 ÷ 年化波动
5. 新增 risk-free（日无风险利率）参数
6. 去掉 csv 对收盘价的依赖，只保留 signal
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import warnings

# ========= 用户可调参数 ========= #
CSV_FILE      = Path("data/raw_data/greed_index_ml.csv")
START_DATE    = "2005-01-03"
TICKER        = "SPY"
FEE_BPS       = 0.0          # **单边**手续费基点；若为 0 表示无手续费
SLIPPAGE_BPS  = 0.0          # 每次换仓滑点
RF_ANNUAL     = 0.00         # 年化无风险利率；0 = 忽略
ANNUAL_FACTOR = 252

PERIODS = [
    ("2005-01-03", "2010-12-31"),
    ("2011-01-01", "2015-12-31"),
    ("2016-01-01", "2020-12-31"),
    ("2021-01-01", "2025-07-07"),
]
# ================================= #

# ---------- 辅助函数 ---------- #
def load_signal(csv_path: Path, start_date: str) -> pd.DataFrame:
    """读取信号，只保留交易日字段"""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.loc[df["Date"] >= start_date, ["Date", "signal"]].copy()
    df.set_index("Date", inplace=True)
    if not {"signal"}.issubset(df.columns):
        raise KeyError("缺少 signal 列")
    return df

def fetch_adj_ohlc(ticker: str, start: str, end: str) -> pd.DataFrame:
    """一次性拉取复权 Open / Close"""
    px = yf.download(
        ticker, start=start, end=end,
        auto_adjust=True, progress=False, actions=False
    )[["Open", "Close"]]
    px.columns = ["adj_open", "adj_close"]
    return px

# ---------- 核心计算 ---------- #
def calc_nav(df_sig: pd.DataFrame,
             px: pd.DataFrame,
             fee_bps: float = 0.0,
             slippage_bps: float = 0.0,
             rf_daily: float = 0.0):
    """
    返回：
        nav_df, strat_ret, bh_ret, trade_days, trade_log
    """
    # === 1) 只保留价格与信号都存在的交易日 ===
    idx = df_sig.index.intersection(px.index)
    if len(idx) < len(df_sig):
        warnings.warn("部分信号日期非交易日，已自动剔除")
    df_sig = df_sig.loc[idx]
    adj_open  = px.loc[idx, "adj_open"]
    adj_close = px.loc[idx, "adj_close"]

    # === 2) 收益拆分 ===
    overnight_ret = (adj_open / adj_close.shift(1) - 1).fillna(0.0)
    intraday_ret  = (adj_close / adj_open - 1).fillna(0.0)

    # === 3) 信号 → 当日目标权重 ===
    map_weight = np.select(
        [df_sig["signal"] == 1,df_sig["signal"] == 2, df_sig["signal"] == 3],
        [0.0, 1.0, 0.8],
        default=np.nan
    )
    weight_today = pd.Series(map_weight, index=df_sig.index).ffill()
    weight_today.iloc[0] = 1.0

    # === 4) 权重生效 (T+1) ===
    w_open = weight_today.shift(1).fillna(1.0)         # 开盘持仓
    w_overnight = w_open.shift(1).fillna(1.0)          # 昨日收盘-今日开盘

    # === 5) 交易成本 ===
    turnover = w_open.diff().abs().fillna(0.0)
    cost = turnover * ((fee_bps + slippage_bps) / 10000.0) * 2   # 双边
    trade_days = int((turnover > 0).sum())

    # === 6) 策略 & 基准日收益 ===
    strat_ret   = w_overnight * overnight_ret + w_open * intraday_ret - cost
    bh_ret      = overnight_ret + intraday_ret
    excess_ret  = strat_ret - rf_daily      # 用于夏普

    # === 7) NAV 累积 ===
    nav = pd.DataFrame({
        "Strategy": (1 + strat_ret).cumprod(),
        "Buy&Hold": (1 + bh_ret).cumprod()
    })
    nav.index.name = "Date"

    # === 8) 逐笔交易日志 ===
    trade_log = _build_trade_log(nav["Strategy"], turnover)

    return nav, strat_ret, bh_ret, excess_ret, trade_days, trade_log

def _build_trade_log(nav_strategy: pd.Series, turnover: pd.Series) -> pd.DataFrame:
    idx_change = turnover[turnover > 0].index
    if idx_change.empty:
        return pd.DataFrame(columns=["Trade_Return"])
    rows, prev_nav = [], nav_strategy.iloc[0]
    for dt in idx_change:
        rows.append({"Date": dt, "Trade_Return": nav_strategy.loc[dt]/prev_nav - 1})
        prev_nav = nav_strategy.loc[dt]
    return pd.DataFrame(rows).set_index("Date")

# ---------- 绩效指标 ---------- #
def perf_summary(nav: pd.Series, daily_ret: pd.Series,
                 excess_ret: pd.Series, annual_factor: int = 252):
    total_ret = nav.iloc[-1] - 1
    years     = (nav.index[-1] - nav.index[0]).days / 365.25
    ann_ret   = (1 + total_ret) ** (1/years) - 1
    ann_vol   = daily_ret.std() * np.sqrt(annual_factor)
    sharpe    = np.nan if ann_vol == 0 else (excess_ret.mean()*annual_factor) / ann_vol
    max_dd    = (nav / nav.cummax() - 1).min()
    return total_ret, ann_ret, ann_vol, sharpe, max_dd

# ---------- 打包回测 ---------- #
def backtest(df_sig: pd.DataFrame, px: pd.DataFrame, label: str = ""):
    nav, strat_r, bh_r, excess_r, trade_days, trade_log = calc_nav(
        df_sig, px, FEE_BPS, SLIPPAGE_BPS, RF_ANNUAL/ANNUAL_FACTOR)
    
    rf_daily   = RF_ANNUAL / ANNUAL_FACTOR    # 与传进 calc_nav 的保持一致
    excess_bh  = bh_r - rf_daily

    # 绩效打印
    metrics = pd.DataFrame(
        {
            "Metric": ["Total Return", "Annualized Return", "Annualized Vol",
                       "Sharpe", "Max Drawdown"],
            "Strategy": perf_summary(nav["Strategy"], strat_r, excess_r),
            "Buy & Hold": perf_summary(nav["Buy&Hold"], bh_r, excess_bh)

        }
    )
    print(f"\n=== Performance Summary {label} ===")
    fmt = lambda x: f"{x:>9.2%}" if abs(x) < 1 else f"{x:>9.3f}"
    print(metrics.to_string(index=False, float_format=fmt))
    print(f"Trades executed: {trade_days}")

    # 逐笔交易
    if trade_log.empty:
        print("-- Trade-by-Trade Returns --  无交易发生")
    else:
        print("\n-- Trade-by-Trade Returns --")
        print(trade_log.to_string(float_format=lambda x: f"{x:.2%}"))

    # Equity Curve
    ax = nav.plot(figsize=(10, 4), title=f"Equity Curve {label}")
    ax.set_ylabel("NAV (Start = 1.0)")
    plt.tight_layout()
    plt.show()

# ---------- 主入口 ---------- #
def main():
    df_sig = load_signal(CSV_FILE, START_DATE)
    # 一次性拉取足够长时间的价格
    px = fetch_adj_ohlc(
        TICKER,
        start=df_sig.index[0].strftime("%Y-%m-%d"),
        end=(df_sig.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    )

    # 全样本
    backtest(df_sig, px, f"(Full) {df_sig.index[0].date()}–{df_sig.index[-1].date()}")

    # 分段
    for s, e in PERIODS:
        mask = (df_sig.index >= s) & (df_sig.index <= e)
        if mask.any():
            backtest(df_sig.loc[mask], px, f"({s} ~ {e})")
        else:
            print(f"[Warning] 区间 {s}~{e} 无数据")

if __name__ == "__main__":
    main()

