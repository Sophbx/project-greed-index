import matplotlib.pyplot as plt
import pandas as pd
import os

def plot_equity_curve(df: pd.DataFrame, equity_col: str = 'cumulative_return', title: str = 'Equity Curve', save_path: str = None):
    """
    Plots the equity curve over time.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df[equity_col], label='Portfolio Value')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return ($)')
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Equity curve saved to {save_path}")
    else:
        plt.show()

def plot_drawdown(df: pd.DataFrame, equity_col: str = 'cumulative_return', title: str = 'Drawdown', save_path: str = None):
    """
    Plots drawdown over time.
    """
    running_max = df[equity_col].cummax()
    drawdown = running_max - df[equity_col]

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, drawdown, color='red', label='Drawdown')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Drawdown ($)')
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Drawdown plot saved to {save_path}")
    else:
        plt.show()
