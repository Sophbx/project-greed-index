#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive plot of close price with two types of signal markers.
"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# 1) CSV 路径（默认同目录）
CSV_PATH = Path("data/raw_data/greed_index_ml.csv")

def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        sys.exit(f"[Error] CSV not found: {csv_path.resolve()}")
    return (
        pd.read_csv(csv_path, parse_dates=["Date"])
          .sort_values("Date")
          .reset_index(drop=True)
    )

def ask_date(label: str, default: str) -> pd.Timestamp:
    s = input(f"{label} [default {default}]: ").strip() or default
    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(dt):
        sys.exit(f"[Error] Invalid date format: {s!r}")
    return dt

def main() -> None:
    df = load_data(CSV_PATH)

    earliest, latest = df["Date"].min().date(), df["Date"].max().date()
    print(f"\nAvailable data: {earliest} → {latest}")

    start_dt = ask_date("Enter START date", str(earliest))
    end_dt   = ask_date("Enter END   date", str(latest))
    if start_dt > end_dt:
        sys.exit("[Error] Start date later than end date.")

    # 2) 过滤区间
    df_sub = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
    if df_sub.empty:
        sys.exit("[Error] No data in selected range.")

    # 3) 绘图
    plt.figure(figsize=(12, 6))
    plt.plot(df_sub["Date"], df_sub["close"], label="Close Price", linewidth=1.2)

    # --- Signal == 1 ---
    sig1 = df_sub[df_sub["signal"] == 1]
    if not sig1.empty:
        plt.scatter(
            sig1["Date"], sig1["close"],
            s=60, marker="o", color="tab:green",
            label="Signal = 1"
        )

    # --- Signal == 2 ---
    sig2 = df_sub[df_sub["signal"] == 2]
    if not sig2.empty:
        plt.scatter(
            sig2["Date"], sig2["close"],
            s=70, marker="^", color="tab:red",
            label="Signal = 2"
        )

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Close Price with Signal Markers\n{start_dt.date()} → {end_dt.date()}")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
