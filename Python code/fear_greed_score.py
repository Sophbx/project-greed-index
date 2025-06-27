import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer, MinMaxScaler
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional, Sequence, Dict
import joblib


def compute_greed_score(df: pd.DataFrame, pipeline: Optional[Pipeline] = None,feature_cols: Optional[Sequence[str]] = None,
                        tech_col: Optional[Sequence[str]] = None, sentiment_col: Optional[Sequence[str]] = None, 
                        flow_col: Optional[Sequence[str]] = None, qt_params: Optional[Dict] = None, 
                        mms_params: Optional[Dict] = None, 
                        weights: Tuple[float, float, float] = (0.4, 0.4, 0.2)) -> Tuple[pd.DataFrame, Pipeline]:
    """
    
    """
    df = df.copy()

#Compute (current volume / volume 20 days moving average) ratio
    df["vol_ratio"] = df["volume"] / df["volume_20d_ma"]
#Inverse VIX and ATR to align them with other indicators, so that they all follow the direction that high value = greed
    df["vix_inv"] = -df["vix_close"]
    df["atr_inv"] = -df["atr"]

#Divide all indicators into groups
    if tech_col is None:
        tech_col = ["rsi", "macd", "macd_signal", "price_zscore", "atr_inv"]
    if sentiment_col is None:
        sentiment_col = ["vix_inv"]
    if flow_col is None:
        flow_col = ["vol_ratio"]
    if feature_cols is None:
        feature_cols = list(tech_col + sentiment_col + flow_col)

#
    if pipeline is None:
        qt_params = {} if qt_params is None else qt_params
        mms_params = {} if mms_params is None else mms_params
        pipeline = Pipeline([("qt", QuantileTransformer(output_distribution = "uniform", **qt_params)), ("mms", MinMaxScaler(**mms_params)), ])
        scaled_arr = pipeline.fit_transform(df[feature_cols])

    else:
        scaled_arr = pipeline.transform(df[feature_cols])
    
    df_scaled = pd.DataFrame(scaled_arr, columns = feature_cols, index = df.index)

#
    df_scaled["tech_sub"] = df_scaled[tech_col].mean(axis = 1)
    df_scaled["sentiment_sub"] = df_scaled[sentiment_col].mean(axis = 1)
    df_scaled["flow_sub"] = df_scaled[flow_col].mean(axis = 1)

#
    tech_weight, sentiment_weight, flow_weight = weights
    greed_raw = (tech_weight * df_scaled["tech_sub"] + sentiment_weight * df_scaled["sentiment_sub"] + flow_weight * df_scaled["flow_sub"])

    greed_score = (greed_raw - greed_raw.min()) / (greed_raw.max() - greed_raw.min())
    df_scaled["greed_index"] = greed_score

    for col in ["tech_sub", "sentiment_sub", "flow_sub", "greed_index"]:
        df[col] = df_scaled[col]
    
    return df, pipeline

#Read the previous generated csv file "Combined_data.csv".
df = pd.read_csv("data/Combined_data.csv", parse_dates = ["Date"], index_col = "Date")

df_with_greed, pipe = compute_greed_score(df)

joblib.dump(pipe, "models/greed_score_pipelinw.joblib")

df_with_greed.to_csv("data/Combined_with_greed.csv")