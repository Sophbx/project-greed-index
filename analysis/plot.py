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

def plot_greed_vs_price(df, price_col='close', greed_col='greed_index', title='Greed Index vs Close Price', save_path=None):
    """
    Plots the Greed Index and Close Price on two y-axes.
    """
    fig, ax1 = plt.subplots(figsize=(12, 5))

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Close Price", color="tab:blue")
    ax1.plot(df.index, df[price_col], label="Close Price", color="tab:blue")
    ax1.tick_params(axis='y', labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Greed Index", color="tab:green")
    ax2.plot(df.index, df[greed_col], label="Greed Index", color="tab:green", alpha=0.7)
    ax2.tick_params(axis='y', labelcolor="tab:green")

    fig.tight_layout()
    plt.title(title)

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Greed vs Price plot saved to {save_path}")
    else:
        plt.show()
