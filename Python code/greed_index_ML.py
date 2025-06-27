
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
# 原始特征
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

# 3. 滞后特征 & 交互项
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
thresh = y_oof_prob.rolling(20).quantile(0.90)
signal = (y_oof_prob > thresh) & (y_oof_prob.shift(1) > thresh.shift(1))

# 7. 输出
out = df.loc[X_all.index, ["close"]].copy()
out["p_pullback"] = y_oof_prob
out["signal"]     = signal.astype(int)
out.to_csv(OUT_FILE)
print(f"已保存信号文件: {OUT_FILE}")








'''
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from greed_fear_index import normalize

#Parameter
DATA_PATH = "data/raw_data/Combined_data_2000.csv"
FEATURE_COL = ["greed_index", "norm_vol_ratio", "norm_atr", "norm_macd_signal"]
HORIZON =  20
DD_THRESH = - 0.1
N_SPLITS = 5
BASE_W = 1.0

CLEAR_PROB = 

FLOOR_W = 0.05
P0 = 0.45
K_STEEP = 10
OUT_FILE = "data/raw_data/greed_index_ml.csv"

#Read previous csv file
df = (pd.read_csv(DATA_PATH, parse_dates = ["Date"]).sort_values("Date").set_index("Date"))

df["vol_ratio"] = df["volume"] / df["volume_20d_ma"]
df["norm_macd_signal"] = normalize(df["macd_signal"])
df["norm_atr"] = normalize(df["atr"])
df["norm_vol_ratio"] = normalize(df["vol_ratio"])

#Define "pullback" label

future_min = (df["close"].shift(-1).rolling(window = HORIZON, min_periods = HORIZON).min().shift(-(HORIZON - 1)))
df["future_min"] = future_min
df["max_drawdown"] = (df["future_min"] - df["close"]) / df["close"]
df["pullback"] = (df["max_drawdown"] <= DD_THRESH).astype(int)
df = df.iloc[:-HORIZON]

#Set the features and label as variables
x = df[FEATURE_COL]
y = df["pullback"]

#Time Series Split and Logistic Regression
tscv = TimeSeriesSplit(n_splits = N_SPLITS)
y_oof_prob = pd.Series(index = y.index, dtype = float)

for train_idx, test_idx in tscv.split(x):
    model = LogisticRegression(class_weight = "balanced", solver = "lbfgs", max_iter = 10000)
    model.fit(x.iloc[train_idx], y.iloc[train_idx])
    y_oof_prob.iloc[test_idx] = model.predict_proba(x.iloc[test_idx])[:, 1]

#Do a simple evaluation
valid = (~y_oof_prob.isna()) & (~y.isna())
pr_auc = average_precision_score(y[valid], y_oof_prob[valid])
print(f"Out of sample PR-AUC = {pr_auc: .3f}")

pos_rate = y.mean()         
print(f"正类比例 = {pos_rate:.3f}")
print("正类(1) 总数:", df["pullback"].sum())
print("样本总数 :", len(df))
print("正类比例 :", df["pullback"].mean())


#Map probability to positions
df.loc[valid.index, "p_pullback"] = y_oof_prob
sigmoid = lambda p: FLOOR_W + (BASE_W - FLOOR_W) / (1 + np.exp(K_STEEP * (p - P0)))
df["target_weight"] = df["p_pullback"].apply(sigmoid)


df[["close"] + FEATURE_COL + ["p_pullback", "target_weight"]].to_csv(OUT_FILE)
print(f"Signal file has saved to {OUT_FILE}")
'''







