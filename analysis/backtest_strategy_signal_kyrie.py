
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal-based position‐sizing back-test
・信号定义：
    0 = no signal    ->  remain previous weight
    1 = signal_risk  ->  0.0
    2 = signal_buy   ->  1.5
    3 = both_signal  ->  1.0
・T+1 调仓，避免偷看未来
・示例未计交易成本；可用 FEE_BPS 调整
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ========= 用户可调参数 ========= #
CSV_FILE   = Path("data/raw_data/greed_index_ml.csv")   # 数据文件路径
FEE_BPS    = 0.0      # 每次换手的双边手续费（基点），例：2 bp = 0.0002
START_DATE = "2005-01-03"
TICKER        = "SPY"
USE_ADJ_CLOSE = True    # True = 自动抓取含分红的 Adj Close
# ================================= #


def load_data(csv_path: Path, start_date: str) -> pd.DataFrame:
    """读取并裁剪信号数据"""
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.loc[df["Date"] >= start_date].copy()
    df.set_index("Date", inplace=True)
    needed_cols = {"close", "signal"}
    if not needed_cols.issubset(df.columns):
        missing = needed_cols - set(df.columns)
        raise KeyError(f"缺少列：{missing}")
    return df

import yfinance as yf

def fetch_adj_close(idx, start, end):
    data = yf.download(idx, start=start, end=end, auto_adjust=True)
    col  = "Adj Close" if "Adj Close" in data.columns else "Close"
    ser  = data[[col]].squeeze("columns")
    ser.name = "adj_close"
    return ser

def calc_nav(df: pd.DataFrame, fee_bps: float, use_adj: bool = True) -> pd.DataFrame:
    """根据信号 -> 权重，计算每日收益、净值"""
    
    if use_adj:
        adj = fetch_adj_close(TICKER, df.index[0].strftime("%Y-%m-%d"),
                              (df.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        price = adj.reindex(df.index).ffill()  # 与信号日期对齐
        if price.isna().any():
            raise ValueError("Adj Close 取值存在缺口，无法对齐日期")
    else:
        if "close" not in df.columns:
            raise KeyError("缺少列：close")
        price = df["close"]

    daily_ret = price.pct_change().fillna(0.0)
    
    '''
    daily_ret = df["close"].pct_change().fillna(0.0)
    '''

    # 信号映射为权重，并 T+1 执行
    mapped = np.select([df["signal"] == 1, df["signal"] == 2, df["signal"] == 3], [0.0, 1.0, 0.8], default = np.nan)
    weight_today = pd.Series(mapped, index = df.index)
    weight_today.iloc[0] = 1.0
    weight_today.ffill(inplace = True)
    '''
    max_flat = 10  # 最长空仓 10 个交易日 (~2 weeks)
    
    flat_counter = 0
    new_w = []

    for w in weight_today:
        if w == 0.0:           # 正在空仓
            flat_counter += 1
            if flat_counter > max_flat:
                w = 1.0        # 超时→回到 1 ×
        else:
            flat_counter = 0   # 非空仓日重置
        new_w.append(w)

    weight_today = pd.Series(new_w, index=df.index)
    '''
    trade_weight = weight_today.shift(1).fillna(1.0)
    # 交易成本：|Δ权重| × fee
    turnover = trade_weight.diff().abs().fillna(0.0)
    cost = turnover * (fee_bps / 10000.0)

    strat_ret = trade_weight * daily_ret - cost
    buyhold_ret = daily_ret

    nav = pd.DataFrame({
        "Strategy": (1 + strat_ret).cumprod(),
        "Buy&Hold": (1 + buyhold_ret).cumprod()
    })
    nav.index.name = "Date"
    return nav, strat_ret, buyhold_ret

def perf_summary(nav: pd.Series, ret: pd.Series, annual_factor: int = 252):
    """总收益、年化收益、波动率、Sharpe、最大回撤"""
    total_return = nav.iloc[-1] - 1
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual_return = (1 + total_return) ** (1 / years) - 1
    annual_vol = ret.std() * np.sqrt(annual_factor)

    sharpe = annual_return / annual_vol if annual_vol != 0 else np.nan
    dd = nav / nav.cummax() - 1
    max_dd = dd.min()
    return total_return, annual_return, annual_vol, sharpe, max_dd


def main():
    df = load_data(CSV_FILE, START_DATE)
    nav, strat_ret, bh_ret = calc_nav(df, FEE_BPS, USE_ADJ_CLOSE)

    # 绩效摘要
    metrics = pd.DataFrame(
        {
            "Metric": ["Total Return", "Annualized Return", "Annualized Vol",
                       "Sharpe", "Max Drawdown"],
            "Strategy": perf_summary(nav["Strategy"], strat_ret),
            "Buy & Hold": perf_summary(nav["Buy&Hold"], bh_ret)
        }
    )
    print("\n=== Performance Summary ({}–{}) ===".format(
        nav.index[0].date(), nav.index[-1].date()))
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:>8.2%}" if abs(x) < 1 else f"{x:>8.3f}"))

    # 画净值曲线
    fig, ax = plt.subplots(figsize=(10, 5))
    nav.plot(ax=ax, title="Equity Curve ({}–Present)".format(START_DATE))
    ax.set_ylabel("NAV (Start = 1.0)")
    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    main()
