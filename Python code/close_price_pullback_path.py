import pandas as pd
import matplotlib.pyplot as plt

def plot_close_and_greed(csv_path: str, start_date: str = None, end_date: str = None):
    """
    """
    df = pd.read_csv(csv_path, parse_dates = [0], index_col = 0)

    if start_date or end_date:
        df = df.loc[start_date: end_date]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df.index, df['close'], color='tab:blue', label='Close Price')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Close Price')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    #
    ax2 = ax1.twinx()
    ax2.plot(df.index, df['p_pullback'], color='tab:orange', label='Pullback Prob')
    ax2.set_ylabel('Pullback Prob (0-1)')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylim(0, 1) 

    # 
    lines, labels = [], []
    for ax in [ax1, ax2]:
        line, label = ax.get_legend_handles_labels()
        lines += line
        labels += label
    ax1.legend(lines, labels, loc='upper left')

    # 
    plt.title('Daily Close Price vs. Pullback Probability')
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    #
    csv_file_path = 'data/raw_data/greed_index_ml.csv'
    s = input("Please enter the start date (YYYY-MM_DD): ").strip()
    e = input("Please enter the end date (YYYY-MM_DD): ").strip()

    plot_close_and_greed(csv_file_path, start_date = s, end_date = e)