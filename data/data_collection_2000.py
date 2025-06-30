import os
import pandas as pd
import numpy as np
import yfinance as yf

from greed_fear_index import normalize, compute_greed_index_simple

# 1. Fetch OHLCV data
def fetch_ohlcv(ticker, start ='2000-01-01', end = '2025-06-27'):
    """
    Download OHLCV data for a given ticker.
    Returns a DataFrame with columns: open, high, low, close, volume.
    """
    df = yf.download(ticker, start = start, end = end, auto_adjust = False)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    return df

# Set up base functions that will be used later to compute technical indicators
def seeded_ema(series: pd.Series, span: int) -> pd.Series:
    '''
    First use the previous span value to get SMA seed
    Then recursively extrapolate EMA
    '''
    ema = series.ewm(span = span, adjust = False, min_periods = span).mean()
    sma_seed = series[:span].mean()
    ema.iloc[span - 1] = sma_seed
    return ema.ffill()

def wilder_rma(series: pd.Series, window: int) -> pd.Series:
    '''
    Wilder method RMA: alpha = 1 / window
    Use SMA(window) seed as the first value
    '''
    rma = series.ewm(alpha = 1 / window, adjust = False, min_periods = window).mean()
    rma.iloc[window - 1] = series[:window].mean()
    return rma.ffill()

# 2. Technical indicator computations
def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower = 0)
    loss = -delta.clip(upper = 0)

    avg_gain = wilder_rma(gain, window)
    avg_loss = wilder_rma(loss, window)

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi
    
def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:

    ema_fast = seeded_ema(series, fast)
    ema_slow = seeded_ema(series, slow)

    macd_line = ema_fast - ema_slow
    signal_line = seeded_ema(macd_line, signal)

    return pd.DataFrame({'macd': macd_line, 'macd_signal': signal_line})

def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis = 1).max(axis = 1)
    atr = wilder_rma(tr, window)

    return atr

def compute_price_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Rolling Z-score of price: (price - rolling_mean) / rolling_std
    """

    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    zscore = (series - mean) / std

    return zscore

# 3. Compute multiple technical indicators
def compute_technical_indicators(df: pd.DataFrame, z_window: int = 20) -> pd.DataFrame:
    tech = pd.DataFrame(index = df.index)
    tech['rsi'] = compute_rsi(df['close'], window = 14)
    macd_df = compute_macd(df['close'], fast = 12, slow = 26, signal = 9)
    tech = tech.join(macd_df)
    tech['atr'] = compute_atr(df['high'], df['low'], df['close'], window = 14)
    tech["price_zscore"] = compute_price_zscore(df["close"], window = z_window)

    return tech

# 4. Assemble all features without dropping rows
def assemble_dataset(start: str = '2000-01-01', end: str = '2025-06-10') -> pd.DataFrame:

    start_dt = pd.to_datetime(start)
    warmup_days = 100  # Safe warm-up buffer
    start_with_buffer = (start_dt - pd.Timedelta(days = warmup_days)).strftime('%Y-%m-%d')

    spy = fetch_ohlcv('SPY', start = start_with_buffer, end = end)
    vix = fetch_ohlcv('^VIX', start = start_with_buffer, end = end)
    spy_tech = compute_technical_indicators(spy)
    vol_20d_ma = spy['volume'].rolling(20).mean().rename("volume_20d_ma")

    df = spy.join(spy_tech, how='inner') \
             .join(vix['close'].rename('vix_close'), how = 'left') \
             .join(vol_20d_ma, how = 'left')
    
    df = df.loc[start:]
    return df

# 5. Basic test suite
def run_tests():
    df_short = fetch_ohlcv('SPY', start = '2020-01-01', end = '2020-01-10')
    assert not df_short.empty, "fetch_ohlcv returned empty DataFrame for short period"
    df = fetch_ohlcv('SPY', start = '2021-01-01', end = '2021-03-01')
    tech = compute_technical_indicators(df, z_window = 20)
    expected = {'rsi', 'macd', 'macd_signal', 'atr', 'price_zscore'}
    missing = expected - set(tech.columns)
    assert not missing, f"Missing technical indicator columns: {missing}"
    
    print("All tests passed.")

if __name__ == '__main__':
    run_tests()
    data = assemble_dataset()

# Normalize
    data['norm_rsi'] = normalize(data['rsi'])
    data['norm_macd'] = normalize(data['macd'])
    data['norm_zscore'] = normalize(data['price_zscore'], -3, 3)
    data['norm_vix'] = 1 - normalize(data['vix_close'])

    # The core index
    data['greed_index'] = compute_greed_index_simple(data)

    os.makedirs('data', exist_ok = True)
    data.to_csv('data/raw_data/Combined_data_2000.csv', index = True)
    print("Data collection and feature assembly complete. CSV saved to 'data/Combined_data.csv'.")



