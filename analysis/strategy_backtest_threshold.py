
"""
strategy_backtest_threshold.py

简单的“满仓 / 清仓”策略 + 回测  
-------------------------------------------------
• 输入:  data/greed_index_ml.csv  
         必须含列: Date, close, p_pullback  
• 规则:  p_pullback > 0.5 → 清仓 (权重=0)  
         p_pullback ≤ 0.5 → 满仓 (权重=1)  
• T+1 生效，手续费单边 fee (默认 5bp)  
• 输出:  data/backtest_threshold_results.csv  
         并在终端打印关键绩效指标
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path

# ----------------- 绩效统计函数 ----------------- #
def perf_stats(r, freq=252, rf=0.0):
    """返回 (年化收益, 年化波动, Sharpe, 最大回撤, 总收益)"""
    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (freq / len(r)) - 1
    ann_vol   = r.std(ddof=0) * np.sqrt(freq)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    peak      = cum.cummax()
    dd        = cum / peak - 1
    max_dd    = dd.min()
    return ann_ret, ann_vol, sharpe, max_dd, total_ret

# ----------------- 核心回测函数 ----------------- #
def run_backtest(path_csv, out_path=None, fee=0.0005):
    df = pd.read_csv(path_csv, parse_dates=['Date']).set_index('Date')
    if 'p_pullback' not in df.columns:
        raise ValueError("输入文件必须包含列 'p_pullback'。")

    # 1. 生成仓位
    df['weight_rule'] = (df['p_pullback'] <= 0.6).astype(float)  # 满仓=1, 清仓=0
    df['weight']      = df['weight_rule'].shift(1).fillna(1.0)   # T+1 生效

    # 2. 收益 & 成本
    df['ret']   = df['close'].pct_change().fillna(0)
    df['gross'] = df['weight'] * df['ret']
    df['tcost'] = df['weight'].diff().abs().fillna(0) * fee
    df['net']   = df['gross'] - df['tcost']

    # 3. 绩效指标
    ann_ret, ann_vol, sharpe, max_dd, total_ret = perf_stats(df['net'])
    ann_ret_bh, ann_vol_bh, sharpe_bh, max_dd_bh, total_ret_bh = perf_stats(df['ret'])

    print("\n=== Threshold Strategy (0.5) ===")
    print(f"Total Return      : {total_ret*100:6.2f}%")
    print(f"Annual Return     : {ann_ret*100:6.2f}%")
    print(f"Annual Volatility : {ann_vol*100:6.2f}%")
    print(f"Sharpe Ratio      : {sharpe:6.2f}")
    print(f"Max Drawdown      : {max_dd*100:6.2f}%")

    print("\n=== Buy & Hold ===")
    print(f"Total Return      : {total_ret_bh*100:6.2f}%")
    print(f"Annual Return     : {ann_ret_bh*100:6.2f}%")
    print(f"Annual Volatility : {ann_vol_bh*100:6.2f}%")
    print(f"Sharpe Ratio      : {sharpe_bh:6.2f}")
    print(f"Max Drawdown      : {max_dd_bh*100:6.2f}%")

    # 4. 保存明细
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=True)
        print(f"\n已保存每日回测结果至: {out_path}")

# ----------------- CLI ----------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Threshold back‑test based on p_pullback.")
    parser.add_argument("--input",  default="data/greed_index_ml.csv", help="输入 CSV 路径")
    parser.add_argument("--output", default="data/backtest_threshold_results.csv", help="输出 CSV 路径")
    parser.add_argument("--fee",    type=float, default=0.0005, help="单边手续费(默认 5bp)")
    args = parser.parse_args()
    run_backtest(args.input, args.output, args.fee)
