# main.py
import os
import pandas as pd
# Decide which data collection python file to use: 1) If want 2000-present, write "from data.data_collection_2000"; 
#                                                  2) If want 2010-present, write "from data.data_collection_2010";
from data.data_collection_2010 import assemble_dataset
from model.strategy_basic import greed_signal
from data.greed_fear_index import compute_greed_index
from analysis.backtest import backtest
from analysis.metrics import evaluate_strategy
from analysis.logger import log_metrics
from analysis.plot import plot_equity_curve, plot_drawdown, plot_greed_vs_price

# 1. Load data
df = assemble_dataset()

# 2. Compute index necessary
df['greed_index'] = compute_greed_index(df)

# 3. Run Strategy
df = greed_signal(df, window=20, price_thresh=0.05)  # output includes 'signal'

# 3*. Run Strategy with ml/rl

# 4. Add data in after recent round of calculation
os.makedirs('data/raw_data', exist_ok=True)
df.to_csv('data/raw_data/Combined_data_2010.csv')

# 5. Backtest
result_df = backtest(df, signal_col='signal')

# 6. Evaluate strategy performance with metrics
metrics = evaluate_strategy(result_df)

# 7. Log the performance in text
log_metrics(metrics, strategy_name='Greed Index Basic')

# 8. Plot the performance in graph
plot_equity_curve(result_df, save_path='logs/basic_strategy_eyquity.png')
plot_drawdown(result_df, save_path='logs/basic_strategy_drawdown.png')
plot_greed_vs_price(result_df, save_path='logs/basic_strategy_greed_close.png')
