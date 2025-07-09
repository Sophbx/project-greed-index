"""
Strategy Summary:
Time to buy: 1. when buy signals occur
             2. when 5 days have passed after last sell
             --> the condition comes first will be executed
Time to sell: 1. when risk signals occur
              2. when there has been a 8% rise in the equity price from last buy
              --> the condition comes first will be executed

Examples: 1. We are at 0 position. A buy signal occur, so we enter the market with 100% position.
             After this buy, we wait for some time but no risk signal occurs yet. During this time,
             we notice that the equity price has risen for 8% compare to our buy price, we immediately
             sell the equity, and our position returns to 0.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Parameters ===
CSV_FILE = Path("data/raw_data/greed_index_ml.csv")
START_DATE = "2005-01-03"
BUY_SIGNAL = 2
RISK_SIGNAL = 1
PRICE_RISE_THRESHOLD = 0.08  
DAYS_AFTER_SELL_REBUY = 5

# === Load Data ===
df = pd.read_csv(CSV_FILE, parse_dates=["Date"])
df = df[df["Date"] >= START_DATE].copy()
df.set_index("Date", inplace=True)

df["position"] = 0
df["signal_trade"] = None

in_position = False
entry_price = 0.0
entry_date = None
last_sell_date = None
trades = []

# === Strategy Logic ===
for i in range(1, len(df)):
    today = df.index[i]
    row = df.iloc[i]
    signal = row["signal"]
    close_price = row["close"]

    if not in_position:
        if signal == BUY_SIGNAL or (
            last_sell_date is not None and (today - last_sell_date).days >= DAYS_AFTER_SELL_REBUY
        ):
            in_position = True
            entry_price = close_price
            entry_date = today
            df.at[today, "position"] = 1
            df.at[today, "signal_trade"] = "buy"
    else:
        price_rise = (close_price - entry_price) / entry_price
        if signal == RISK_SIGNAL or price_rise >= PRICE_RISE_THRESHOLD:
            in_position = False
            df.at[today, "position"] = 0
            df.at[today, "signal_trade"] = "sell"
            last_sell_date = today
            trades.append((entry_date, today, entry_price, close_price))

    if in_position and pd.isna(df.at[today, "position"]):
        df.at[today, "position"] = 1

# === Analyze Trades ===
trade_df = pd.DataFrame(trades, columns=["Buy Date", "Sell Date", "Buy Price", "Sell Price"])
trade_df["Return %"] = (trade_df["Sell Price"] - trade_df["Buy Price"]) / trade_df["Buy Price"] * 100

df["portfolio_value"] = np.nan
df["hold_value"] = df["close"] / df["close"].iloc[0]

current_value = 1.0
in_position = False
entry_price = 0.0

for i in range(len(df)):
    date = df.index[i]
    row = df.iloc[i]
    signal = row.get("signal_trade", None)
    close = row["close"]

    if signal == "buy":
        in_position = True
        entry_price = close
    elif signal == "sell" and in_position:
        ret = (close - entry_price) / entry_price
        current_value *= (1 + ret)
        in_position = False

    df.at[date, "portfolio_value"] = current_value

# === Metrics ===
df["portfolio_value"].ffill(inplace=True)
df["portfolio_value"].fillna(method="bfill", inplace=True)

# === Daily Strategy Returns
df["strategy_daily_return"] = df["portfolio_value"].pct_change()
df["hold_daily_return"] = df["hold_value"].pct_change()

# === Strategy Metrics
strategy_total_return = df["portfolio_value"].iloc[-1] - 1.0
# Calculate Annual Return for Strategy
n_days = df["portfolio_value"].count()
strategy_annual_return = (df["portfolio_value"].iloc[-1])**(252 / n_days) - 1
strategy_sharpe = (
    df["strategy_daily_return"].mean() / df["strategy_daily_return"].std() * np.sqrt(252)
    if df["strategy_daily_return"].std() > 0 else 0
)
peak = df["portfolio_value"].cummax()
dd = (df["portfolio_value"] - peak) / peak
strategy_max_drawdown = dd.min()

# === Buy & Hold Metrics
hold_total_return = df["hold_value"].iloc[-1] - 1.0
# Calculate Annual Return for Buy & Hold
hold_annual_return = (df["hold_value"].iloc[-1])**(252 / n_days) - 1
hold_sharpe = (
    df["hold_daily_return"].mean() / df["hold_daily_return"].std() * np.sqrt(252)
    if df["hold_daily_return"].std() > 0 else 0
)
peak_hold = df["hold_value"].cummax()
dd_hold = (df["hold_value"] - peak_hold) / peak_hold
hold_max_drawdown = dd_hold.min()

# === Print Clean Results
print("==== Strategy Performance ====")
print(f"Total Return: {strategy_total_return:.2%}")
print(f"Annual Return: {strategy_annual_return:.2%}")
print(f"Sharpe Ratio: {strategy_sharpe:.2f}")
print(f"Max Drawdown: {strategy_max_drawdown:.2%}")
print("\n==== Buy & Hold Performance ====")
print(f"Total Return: {hold_total_return:.2%}")
print(f"Annual Return: {hold_annual_return:.2%}")
print(f"Sharpe Ratio: {hold_sharpe:.2f}")
print(f"Max Drawdown: {hold_max_drawdown:.2%}")

# === Build Portfolio Value for Strategy vs. Buy & Hold ===
df["portfolio_value"] = np.nan
df["hold_value"] = df["close"] / df["close"].iloc[0]  # normalized

current_value = 1.0
in_position = False
entry_price = 0.0

for i in range(len(df)):
    date = df.index[i]
    row = df.iloc[i]
    signal = row["signal_trade"]
    close = row["close"]

    if signal == "buy":
        in_position = True
        entry_price = close
    elif signal == "sell" and in_position:
        ret = (close - entry_price) / entry_price
        current_value *= (1 + ret)
        in_position = False
    df.at[date, "portfolio_value"] = current_value

df["portfolio_value"].ffill(inplace=True)
df["portfolio_value"].fillna(method="bfill", inplace=True)

# === Plot Performance ===
plt.figure(figsize=(14, 6))
plt.plot(df.index, df["portfolio_value"], label="Strategy Portfolio")
plt.plot(df.index, df["hold_value"], label="Buy & Hold", linestyle="--")
plt.title("Strategy vs Buy & Hold Performance")
plt.xlabel("Date")
plt.ylabel("Normalized Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()# === Build Portfolio Value for Strategy vs. Buy & Hold ===
df["portfolio_value"] = np.nan
df["hold_value"] = df["close"] / df["close"].iloc[0]  # normalized