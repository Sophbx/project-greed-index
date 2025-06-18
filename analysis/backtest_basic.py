import pandas as pd

def backtest(df: pd.DataFrame, signal_col: str = 'signal', initial_cash: float = 100000) -> pd.DataFrame:
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
