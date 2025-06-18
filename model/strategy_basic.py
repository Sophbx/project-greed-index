import pandas as pd

def generate_greed_index_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Adds trading signals based on comparison between current greed index and its rolling mean.
    Use the standard window period of 20 days here.
    Signal:
        1 = Long
       -1 = Short
        0 = Hold
    """
    df = df.copy()
    
    df['greed_mean'] = df['greed_index'].rolling(window).mean()

    def classify(row):
        if pd.isna(row['greed_mean']):
            return 0  # avoid early NaNs
        if row['greed_index'] > 1.1 * row['greed_mean']:
            return -1  # short
        elif row['greed_index'] < 0.9 * row['greed_mean']:
            return 1   # long
        else:
            return 0   # hold

    df['signal'] = df.apply(classify, axis=1)

    return df
