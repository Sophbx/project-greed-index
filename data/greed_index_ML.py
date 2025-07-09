
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import PolynomialFeatures

from greed_fear_index import normalize

# 参数
DATA_PATH   = "data/raw_data/Combined_data_2000.csv"
HORIZON     = 20
DD_THRESH   = -0.1
N_SPLITS    = 5
OUT_FILE    = "data/raw_data/greed_index_ml.csv"

# 1. 读取与初步特征
df = (
    pd.read_csv(DATA_PATH, parse_dates=["Date"])
      .sort_values("Date")
      .set_index("Date")
)
df["vol_ratio"]       = df["volume"] / df["volume_20d_ma"]
df["norm_macd_signal"]= normalize(df["macd_signal"])
df["norm_atr"]        = normalize(df["atr"])
df["greed_index"]     = df["greed_index"]

# 2. 构造标签：未来 HORIZON 天内最大回撤
future_min = (
    df["close"]
      .shift(-1)
      .rolling(window = HORIZON, min_periods = HORIZON)
      .min()
      .shift(-(HORIZON - 1))
)
df["max_drawdown"] = (future_min - df["close"]) / df["close"]
df["pullback"]     = (df["max_drawdown"] <= DD_THRESH).astype(int)
df = df.iloc[:-HORIZON]

# 3. logistic regression model训练用的滞后特征 & 交互项
base_feats = ["greed_index", "vol_ratio", "norm_macd_signal", "norm_atr"]
for feat in base_feats:
    df[f"{feat}_lag1"] = df[feat].shift(1)
    df[f"{feat}_lag2"] = df[feat].shift(2)

# 准备多项式交互
feat_lags = [f"{f}_lag{l}" for f in base_feats for l in [1,2]]
df_feats   = df[feat_lags].dropna()
poly       = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_all      = pd.DataFrame(
    poly.fit_transform(df_feats),
    index=df_feats.index,
    columns=poly.get_feature_names_out(df_feats.columns)
)
y_all = df.loc[X_all.index, "pullback"]

# 4. 时间序列 CV + 正则化调参
tscv       = TimeSeriesSplit(n_splits=N_SPLITS)
lr         = LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=5000)
param_grid = {"C": [0.01, 0.1, 1, 10]}
gsearch    = GridSearchCV(lr, param_grid, cv=tscv, scoring="average_precision", n_jobs=-1)
gsearch.fit(X_all, y_all)
best_lr    = gsearch.best_estimator_

# 5. OOF 概率 & 动态阈值信号
y_oof_prob = pd.Series(index=X_all.index, dtype=float)
for train_idx, test_idx in tscv.split(X_all):
    clf = best_lr.set_params()  # 用相同参数做每折验证
    clf.fit(X_all.iloc[train_idx], y_all.iloc[train_idx])
    y_oof_prob.iloc[test_idx] = clf.predict_proba(X_all.iloc[test_idx])[:,1]

# 评估
valid_idx = ~y_oof_prob.isna()
pr_auc    = average_precision_score(y_all[valid_idx], y_oof_prob[valid_idx])
print(f"OOF PR-AUC = {pr_auc:.3f}")


pos_rate = y_all.mean()         
print(f"正类比例 = {pos_rate:.3f}")
print("正类(1) 总数:", df["pullback"].sum())
print("样本总数 :", len(df))
print("正类比例 :", df["pullback"].mean())

# 6. 动态阈值 & 信号平滑
# Dynamic Quantile restriction
thresh_high = y_oof_prob.rolling(20).quantile(0.90) # 过去20天内的数据的前10%线
thresh_low = y_oof_prob.rolling(20).quantile(0.05) # 过去20天内的数据的后5%线

# price restriction
price_3d_ago = df["close"].shift(3)
price_increase_3d = (df["close"] - price_3d_ago) / price_3d_ago
risk_condition = price_increase_3d >= 0.05 # price rise for more than 5% in the past 3 days

# Greed Index restriction
greed_max_20 = df["greed_index"].shift(1).rolling(window = 20).max()
greed_high_bool = (df["greed_index"] >= 0.6).astype(int)
greed_low_bool  = (df["greed_index"] <= 0.45).astype(int)
consec_high = greed_high_bool.rolling(window = 10, min_periods = 10).sum() == 10 # greed index >=0.6 for the past 10 consecutive days
consec_low  = greed_low_bool .rolling(window = 6, min_periods = 6).sum() == 6 # greed index <=0.45 for the past 6 consecutive days

# Market Overheated:
# 1. pullback probab > thresh_high for 2 days; 2. consec_high occurs; 3. risk_condition occurs
signal_risk = (y_oof_prob > thresh_high) & (y_oof_prob.shift(1) > thresh_high.shift(1)) & consec_high.loc[X_all.index]

'''
& risk_condition.loc[X_all.index]
'''

signal_buy = (df.loc[X_all.index, "greed_index"] < (greed_max_20.loc[X_all.index] - 0.29)) & consec_low.loc[X_all.index]

both_signal = signal_risk & signal_buy

signal = np.select([both_signal, signal_risk, signal_buy], [3, 1, 2], default = 0).astype(int)

# 7. 输出
out = df.loc[X_all.index, ["close", "greed_index"]].copy()
out["p_pullback"] = y_oof_prob
out["signal"]     = signal
# Merge the two columns back into the full dataset
df_main = pd.read_csv(OUT_FILE, parse_dates=["Date"])
df_main.set_index("Date", inplace=True)

# Make sure index matches
out.index.name = "Date"

# Drop existing columns if they already exist
df_main.drop(columns=["p_pullback", "signal"], errors="ignore", inplace=True)

# Join and save
df_updated = df_main.join(out[["p_pullback", "signal"]])
df_updated.to_csv(OUT_FILE)
print("[UPDATED] Columns 'p_pullback' and 'signal' merged into Combined_data_2000.csv")
print(f"Signal by ML saved.")
