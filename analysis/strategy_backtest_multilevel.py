"""
多档仓位 (bucketed weights) 策略 + 回测
-------------------------------------------------
• 输入:  data/greed_index_ml.csv  (必须含 Date, close, p_pullback)
• 默认权重映射:
    p_pullback ≤ 0.25  -> 1.0   (满仓)
    0.25 < p ≤ 0.50    -> 0.6
    0.50 < p ≤ 0.75    -> 0.3
    p_pullback  > 0.75 -> 0.0   (空仓)
• CLI 可自定义 --bins 与 --weights
• T+1 生效，手续费单边 5bp (可改 --fee)
• 输出: data/backtest_multilevel_results.csv
"""

import pandas as pd
import numpy as np
import argparse
from ast import literal_eval
from pathlib import Path

# ----------------- 绩效统计 ----------------- #
def perf_stats(r, freq=252, rf=0.0):
    cum = (1 + r).cumprod()
    total_ret = cum.iloc[-1] - 1
    ann_ret   = (1 + total_ret) ** (freq / len(r)) - 1
    ann_vol   = r.std(ddof=0) * np.sqrt(freq)
    sharpe    = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    peak      = cum.cummax()
    dd        = cum / peak - 1
    max_dd    = dd.min()
    return ann_ret, ann_vol, sharpe, max_dd, total_ret

# ----------------- 主函数 ----------------- #
def run_backtest(path_csv, bins, weights, out_path=None, fee=0.0005, rebalance='D'):
    if len(bins) - 1 != len(weights):
        raise ValueError("len(weights) 必须等于 len(bins)-1")
    df = pd.read_csv(path_csv, parse_dates=['Date']).set_index('Date')
    if 'p_pullback' not in df.columns:
        raise ValueError("输入文件必须包含列 'p_pullback'")

    # 1. 原始日级 weight_rule
    df['weight_rule'] = pd.cut(df['p_pullback'], bins=bins, labels=weights).astype(float)

    # 2. 可选重采样 (如 W-FRI) 先聚合后 forward-fill
    if rebalance.upper() != 'D':
        rule = df['weight_rule'].resample(rebalance).last()
        df['weight_rule'] = rule.reindex(df.index).ffill()

    # 3. T+1 生效
    df['weight'] = df['weight_rule'].shift(1).fillna(1.0)

    # 4. 收益 & 成本
    df['ret']   = df['close'].pct_change().fillna(0)
    df['gross'] = df['weight'] * df['ret']
    df['tcost'] = df['weight'].diff().abs().fillna(0) * fee
    df['net']   = df['gross'] - df['tcost']

    # 5. 绩效指标
    ann_ret, ann_vol, sharpe, max_dd, total_ret = perf_stats(df['net'])
    ann_ret_bh, ann_vol_bh, sharpe_bh, max_dd_bh, total_ret_bh = perf_stats(df['ret'])

    print("\n=== Multi-Level Weight Strategy ===")
    print("Bins    :", bins)
    print("Weights :", weights)
    if rebalance.upper() != 'D':
        print("Rebalance freq :", rebalance)
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

    # 6. 保存
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=True)
        print(f"\n已保存回测明细至: {out_path}")

# ----------------- CLI ----------------- #
def str_to_list(s):
    try:
        return list(literal_eval(s))
    except Exception as e:
        raise argparse.ArgumentTypeError("请用 python list 字面量，例如 \"[0,0.25,0.5,0.75,1]\"")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-level weight back-test")
    parser.add_argument("--input",  default="data/greed_index_ml.csv", help="输入 CSV 路径")
    parser.add_argument("--output", default="data/backtest_multilevel_results.csv", help="输出 CSV 路径")
    parser.add_argument("--bins",   type=str_to_list, default="[0,0.6,1]",
                        help="分桶边界 list，长度 n+1 (例: \"[0,0.25,0.5,0.75,1]\")")
    parser.add_argument("--weights",type=str_to_list, default="[1,0]",
                        help="各桶对应权重 list，长度 n (例: \"[1,0.6,0.3,0]\")")
    parser.add_argument("--fee",    type=float, default=0, help="单边手续费(默认 5bp)")
    parser.add_argument("--rebalance", default="D", help="调仓频率: D(默认)/W-FRI/M(月末)")
    args = parser.parse_args()

    run_backtest(args.input, args.bins, args.weights, args.output, args.fee, args.rebalance)
