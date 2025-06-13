import pandas as pd

def normalize(series: pd.Series, low=None, high=None) -> pd.Series:
    """Normalize a series to [0, 1] based on min/max or provided bounds."""
    if low is None: low = series.min()
    if high is None: high = series.max()
    return ((series - low) / (high - low)).clip(0, 1)

def compute_greed_index(df: pd.DataFrame) -> pd.Series:
    """
    Compute a custom greed index from technical indicators.
    Assumes df contains: rsi, macd, price_zscore, vix_close
    """
    norm_rsi = normalize(df['rsi'])
    norm_macd = normalize(df['macd'])
    norm_zscore = normalize(df['price_zscore'])
    norm_vix = 1 - normalize(df['vix_close'])  # invert: high VIX → fear → low greed

    # The weight of each indicator can be tune: compare the performance of the strategies by 
    # adjusting this core index
    greed_index = (
        0.4 * norm_rsi +
        0.2 * norm_macd +
        0.2 * norm_zscore +
        0.2 * norm_vix
    ).clip(0, 1)

    return greed_index

def compute_fear_index(greed_index: pd.Series) -> pd.Series:
    """Fear is simply the inverse of greed on a 0–1 scale."""
    return 1 - greed_index
