import pandas as pd

def backtest_buy_sell_strategy(df: pd.DataFrame) -> pd.DataFrame:
    trades = []
    position = None  # Track if we are currently in a trade
    buy_price = None
    buy_date = None

    for current_date, row in df.iterrows():
        signal = row["signal"]
        close_price = row["close"]

        if position is None:
            # Look for a buy signal (signal == 2)
            if signal == 2:
                position = "long"
                buy_price = close_price
                buy_date = current_date
        else:
            # Already in a position, look for sell signal (signal == 1) and profit
            if signal == 1 and close_price > buy_price:
                trades.append({
                    "buy_date": buy_date,
                    "sell_date": current_date,
                    "buy_price": buy_price,
                    "sell_price": close_price,
                    "profit_pct": (close_price - buy_price) / buy_price * 100
                })
                # Reset
                position = None
                buy_price = None
                buy_date = None

    return pd.DataFrame(trades)

def compute_return_series(df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    df['exec_signal'] = 0
    df['strategy_return'] = 0.0
    df['cumulative_return'] = 1.0

    for _, trade in trades.iterrows():
        buy_date = trade['buy_date']
        sell_date = trade['sell_date']
        buy_price = trade['buy_price']
        sell_price = trade['sell_price']

        if buy_date in df.index and sell_date in df.index:
            df.loc[buy_date, 'exec_signal'] = 1
            df.loc[sell_date, 'exec_signal'] = -1
            ret = (sell_price - buy_price) / buy_price
            df.loc[sell_date, 'strategy_return'] = ret

    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    return df

from metrics import evaluate_strategy

if __name__ == "__main__":
    df = pd.read_csv("data/raw_data/Combined_data_NVDA_2010.csv", parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    trades = backtest_buy_sell_strategy(df)
    df = compute_return_series(df, trades)

    metrics = evaluate_strategy(df)
    for k, v in metrics.items():
        print(f"{k:<15}: {v:.4f}")

