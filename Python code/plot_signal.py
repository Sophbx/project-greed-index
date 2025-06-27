#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive plot of close price with signal markers.

Run:  python plot_signal.py
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1) 设置 CSV 路径（默认同目录）
# ----------------------------------------------------------------------
CSV_PATH = Path("data/raw_data/greed_index_ml.csv")

# ----------------------------------------------------------------------
def load_data(csv_path: Path) -> pd.DataFrame:
    """读取 CSV，确保包含 Date, close, signal 三列。"""
    if not csv_path.exists():
        sys.exit(f"[Error] CSV not found: {csv_path.resolve()}")
    return (
        pd.read_csv(csv_path, parse_dates=["Date"])
          .sort_values("Date")
          .reset_index(drop=True)
    )

def ask_date(prompt: str, fallback: str) -> pd.Timestamp:
    """交互式获取日期；留空则返回 fallback。"""
    s = input(f"{prompt} [default {fallback}]: ").strip()
    if not s:
        s = fallback
    dt = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    if pd.isna(dt):
        sys.exit(f"[Error] Invalid date format: {s!r} (use YYYY-MM-DD)")
    return dt

def main() -> None:
    df = load_data(CSV_PATH)

    earliest = df["Date"].min().date()
    latest   = df["Date"].max().date()
    print(f"\nAvailable data range: {earliest} → {latest}")

    # ------------------------------------------------------------------
    # 2) 询问日期区间
    # ------------------------------------------------------------------
    start_dt = ask_date("Enter START date", str(earliest))
    end_dt   = ask_date("Enter END   date", str(latest))

    if start_dt > end_dt:
        sys.exit("[Error] Start date is later than end date.")

    # ------------------------------------------------------------------
    # 3) 过滤数据并绘图
    # ------------------------------------------------------------------
    mask = (df["Date"] >= start_dt) & (df["Date"] <= end_dt)
    df_sub = df.loc[mask]
    if df_sub.empty:
        sys.exit("[Error] No data in the selected date range.")

    plt.figure(figsize=(12, 6))
    plt.plot(df_sub["Date"], df_sub["close"], label="Close Price", linewidth=1.2)

    sig = df_sub[df_sub["signal"] == 1]
    if not sig.empty:
        plt.scatter(sig["Date"], sig["close"], s=60, marker="o", label="Signal = 1")

    plt.xlabel("Date")
    plt.ylabel("Close Price")
    plt.title(f"Close Price with Signal Markers\n{start_dt.date()} → {end_dt.date()}")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
