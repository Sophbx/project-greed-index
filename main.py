# main.py
import os
import pandas as pd
from data.data_collection_2010 import assemble_dataset
from model.strategy_basic import generate_greed_index_signals
from data.greed_fear_index import compute_greed_index
from analysis.backtest_basic import backtest
from analysis.metrics import evaluate_strategy
from analysis.logger import log_metrics
from analysis.plot import plot_equity_curve, plot_drawdown

# Step 1: Load and prepare data
df = assemble_dataset()
df['greed_index'] = compute_greed_index(df)

# Step 2: Generate signals
df = generate_greed_index_signals(df, window=20)  # output includes 'signal'

# Step 3: Save signals for inspection
os.makedirs('data/raw_data', exist_ok=True)
df.to_csv('data/raw_data/greed_strategy_signals.csv')

# Step 4: Backtest
result_df = backtest(df, signal_col='signal')

# Step 5: Evaluate performance
metrics = evaluate_strategy(result_df)

# Step 6: Log performance
log_metrics(metrics, strategy_name='Greed Index Basic')

# Step 7: Plot results
plot_equity_curve(result_df, save_path='logs/greed_strategy_equity.png')
plot_drawdown(result_df, save_path='logs/greed_strategy_drawdown.png')
