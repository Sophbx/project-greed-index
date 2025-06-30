import pandas as pd

def backtest_basic(df: pd.DataFrame, signal_col: str = 'signal', initial_cash: float = 100000) -> pd.DataFrame:
    """
    Simulates basic long/short trading strategy:
    - Enter at next day open
    - Exit at next day close
    """
    df = df.copy()

    # Shift signal to act at next day open
    df['exec_signal'] = df[signal_col].shift(1)

    # Simulate entry/exit prices
    df['next_open'] = df['open'].shift(-1)
    df['next_close'] = df['close'].shift(-1)

    df['strategy_return'] = 0.0

    long_mask = df['exec_signal'] == 1
    short_mask = df['exec_signal'] == -1

    df.loc[long_mask, 'strategy_return'] = (
        (df.loc[long_mask, 'next_close'] - df.loc[long_mask, 'next_open']) / df.loc[long_mask, 'next_open']
    )
    df.loc[short_mask, 'strategy_return'] = (
        (df.loc[short_mask, 'next_open'] - df.loc[short_mask, 'next_close']) / df.loc[short_mask, 'next_open']
    )

    df['cumulative_return'] = (1 + df['strategy_return'].fillna(0)).cumprod() * initial_cash

    return df

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
