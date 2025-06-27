import pandas as pd

def greed_signal(df: pd.DataFrame, window: int = 20, price_thresh: float = 0.05) -> pd.DataFrame:
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
    df['fear_mean'] = 1 - df['greed_index'].rolling(window).mean()
    df['up_streak'] = (df['close'].diff() > 0).rolling(3).sum() == 3
    df['down_streak'] = (df['close'].diff() < 0).rolling(3).sum() == 3
    df['price_3d_change'] = df['close'] / df['close'].shift(3) - 1

    def signal(row):
        if pd.isna(row['greed_mean']):
            return 0  # avoid early NaNs
        if (row['greed_index'] >= 1.1 * row['greed_mean']) and (row['up_streak']):
            if row['price_3d_change'] >= price_thresh:
                return -1 # SHORT
            else:
                return 1 # LONG: not the end for rise
        elif (row['greed_index'] <= 0.9 * row['fear_mean']) and (row['down_streak']):
            if row['price_3d_change'] >= price_thresh:
                return 1 # LONG
            else:
                return -1 # SHORT: not the end for drop
        else:
            return 0   # holds

    df['signal'] = df.apply(signal, axis=1)

    return df
