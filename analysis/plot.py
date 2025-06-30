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

def plot_close_greed_pullback_and_signal(csv_path: str, start_date: str = None, end_date: str = None, save_path=None):
    """
    Used to plot the close price and pullback probability of an input period of time.
    Observes the relationship between close price + greed index and pullback.
    """
    # Read the targeted csv data file from the start (first row, first column)
    df = pd.read_csv(csv_path, parse_dates = ['Date'])
    df.set_index("Date", inplace=True)

    if start_date or end_date:
        df = df.loc[start_date: end_date]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['close'], color='tab:blue', label='Close Price', linewidth=1.5)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    # Add signal markers
    if 'signal' in df.columns:
        sig1 = df[df['signal'] == 1]
        if not sig1.empty:
            ax1.scatter(sig1.index, sig1['close'], color='tab:red', s=60, marker='^', label='Signal = short')
        sig2 = df[df['signal'] == 2]
        if not sig2.empty:
            ax1.scatter(sig2.index, sig2['close'], color='tab:green', s=70, marker='o', label='Signal = long')
        # sig3 = df[df['signal'] == 3]
        # if not sig3.empty:
            # ax1.scatter(sig3.index, sig3['close'], color='tab:orange', s=80, marker='*', label='Signal = both')

    #
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['greed_index'], color='tab:purple', label='Greed Score', alpha=0.7)
    ax2.plot(df.index, df['p_pullback'], color='black', label='Pullback Prob', alpha=0.7)
    ax2.set_ylabel('Greed / Pullback Prob (0-1)')
    ax2.set_ylim(0, 1) 
    ax2.tick_params(axis='y', labelcolor='tab:orange')

    # 
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    ax1.legend(lines, labels, loc='upper left')

    # 
    plt.title('Daily Close Price, Greed Score, Pullback Probability, & Signals')
    fig.tight_layout()
    plt.show()

    if save_path:
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Greed vs Price plot saved to {save_path}")
    else:
        plt.show()
