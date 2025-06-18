import pandas as pd
import numpy as np

def evaluate_strategy(df: pd.DataFrame, return_col: str = 'strategy_return', equity_col: str = 'cumulative_return') -> dict:
    """
    Computes key metrics for a backtested strategy.
    Returns a dictionary of metrics.
    """
    results = {}
    
    returns = df[return_col].dropna()
    equity = df[equity_col].dropna()

    # Sharpe ratio (assumes daily returns)
    results['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else np.nan

    # Max drawdown
    running_max = equity.cummax()
    drawdown = running_max - equity
    results['max_drawdown'] = drawdown.max()

    # Total return
    results['total_return'] = equity.iloc[-1] / equity.iloc[0] - 1

    # Win rate
    results['win_rate'] = (returns > 0).mean()

    # Number of trades
    results['num_trades'] = df['exec_signal'].abs().sum()

    return results
