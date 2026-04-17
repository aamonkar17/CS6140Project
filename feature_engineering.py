"""
CS 6140 - Hull Tactical Market Prediction
Feature Engineering Script — v2 

Reads preprocessed train/test from pipelines and builds a richer
feature space before modelling. Nine feature blocks are applied in sequence,
followed by a correlation-based selection step to remove noise.
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew
import lightgbm as lgb

warnings.filterwarnings("ignore")

INPUT_DIR  = "outputs"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_COL = "market_forward_excess_returns"
ID_COLS    = ["row_id", "date_id"]
LEAKAGE    = ["forward_returns", "risk_free_rate"]
EPS        = 1e-8

# feature family prefixes and their column indices
# E=economic, I=institutional, M=market, P=price, S=sentiment, V=volatility
PREFIXES = {
    "E": list(range(1, 21)),
    "I": list(range(1, 10)),
    "M": list(range(1, 19)),
    "P": list(range(1, 14)),
    "S": list(range(1, 13)),
    "V": list(range(1, 14)),
}


# HELPERS

def get_cols(df, prefix, nums):
    # returns only columns that are present in df — handles dropped cols safely
    return [f"{prefix}{i}" for i in nums if f"{prefix}{i}" in df.columns]

def load_preprocessed():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train_preprocessed.csv"))
    test  = pd.read_csv(os.path.join(INPUT_DIR, "test_preprocessed.csv"))
    print(f"  Loaded train : {train.shape}")
    print(f"  Loaded test  : {test.shape}")
    return train, test


# BLOCK 1 — GROUP STATS
# Computes summary statistics across each feature family per row.
# Skewness captures distributional asymmetry within the family, and CV
# (std / mean) measures relative dispersion independent of scale.

def block_group_stats(df):
    df = df.copy()
    for pfx, nums in PREFIXES.items():
        cols = get_cols(df, pfx, nums)
        if not cols:
            continue
        g = df[cols]
        df[f"{pfx}_mean"]   = g.mean(axis=1)
        df[f"{pfx}_std"]    = g.std(axis=1).fillna(0)
        df[f"{pfx}_min"]    = g.min(axis=1)
        df[f"{pfx}_max"]    = g.max(axis=1)
        df[f"{pfx}_range"]  = g.max(axis=1) - g.min(axis=1)
        df[f"{pfx}_median"] = g.median(axis=1)
        df[f"{pfx}_skew"]   = g.apply(
            lambda r: skew(r.dropna()) if r.dropna().shape[0] > 2 else 0, axis=1
        )
        df[f"{pfx}_cv"]     = df[f"{pfx}_std"] / (df[f"{pfx}_mean"].abs() + EPS)
    print(f"  [1] Group stats → {df.shape[1]} cols")
    return df


# BLOCK 2 — CROSS-SECTIONAL RANK FEATURES
# Ranks each feature value within its family per row on a 0-1 percentile scale.
# Rank-based features are robust to outliers and distributional shifts over time,
# which are common in financial time-series data.

def block_rank_features(df):
    df = df.copy()
    for pfx, nums in PREFIXES.items():
        cols = get_cols(df, pfx, nums)
        if not cols:
            continue
        ranked = df[cols].rank(axis=1, pct=True)
        df[f"{pfx}_top_rank"]    = ranked.max(axis=1)
        df[f"{pfx}_bot_rank"]    = ranked.min(axis=1)
        df[f"{pfx}_rank_spread"] = ranked.max(axis=1) - ranked.min(axis=1)
        df[f"{pfx}_n_above_med"] = (ranked > 0.5).sum(axis=1)  # majority signal direction
    print(f"  [2] Rank features → {df.shape[1]} cols")
    return df


# BLOCK 3 — Z-SCORE FEATURES PER ROW
# Standardizes each family's values within a row to capture relative strength.
# zsum_pos and zsum_neg separate the positive and negative signal contributions,
# which is more informative than the net sum alone.

def block_zscore_features(df):
    df = df.copy()
    for pfx, nums in PREFIXES.items():
        cols = get_cols(df, pfx, nums)
        if not cols:
            continue
        g   = df[cols]
        mu  = g.mean(axis=1)
        sig = g.std(axis=1).replace(0, EPS)
        df[f"{pfx}_zscore_mean"] = (mu - mu.mean()) / (mu.std() + EPS)
        z = g.sub(mu, axis=0).div(sig, axis=0)
        df[f"{pfx}_zsum"]     = z.sum(axis=1)
        df[f"{pfx}_zsum_pos"] = z.clip(lower=0).sum(axis=1)
        df[f"{pfx}_zsum_neg"] = z.clip(upper=0).sum(axis=1)
    print(f"  [3] Z-score features → {df.shape[1]} cols")
    return df


# BLOCK 4 — VOLATILITY-NORMALIZED SIGNALS
# Divides each signal family by V-family (volatility) to produce risk-adjusted
# features. This is directly aligned with the Sharpe ratio evaluation metric,
# since Sharpe = mean return / volatility. IR_M and IR_E are information
# ratio proxies for the market and economic signal families respectively.

def block_vol_normalized(df):
    df  = df.copy()
    if "V_mean" not in df.columns:
        print("  [4] Skipped — V_mean not found")
        return df

    vol = df["V_mean"].abs() + EPS

    for pfx in ["E", "M", "P", "S", "I"]:
        mc = f"{pfx}_mean"
        rc = f"{pfx}_range"
        if mc in df.columns:
            df[f"{pfx}_vol_adj"]   = df[mc] / vol
            df[f"{pfx}_vol_ratio"] = df[mc] / (df["V_mean"] + EPS)
        if rc in df.columns:
            df[f"{pfx}_range_vol"] = df[rc] / vol

    if "M_mean" in df.columns:
        df["IR_M"] = df["M_mean"] / vol
    if "E_mean" in df.columns:
        df["IR_E"] = df["E_mean"] / vol
    if "P_mean" in df.columns:
        df["momentum_sharpe"] = df["P_mean"] / (df.get("V_std", vol) + EPS)

    print(f"  [4] Vol-normalized → {df.shape[1]} cols")
    return df


# BLOCK 5 — REGIME-CONDITIONED FEATURES
# D-cols are binary regime flags: 1=bullish, -1=bearish, 0=neutral.
# Multiplying continuous signals by D_net allows the model to learn that
# the same economic or market signal can carry different implications
# depending on the prevailing market regime.

def block_regime_conditioned(df):
    df = df.copy()
    d_cols = [f"D{i}" for i in range(1, 10) if f"D{i}" in df.columns]
    if not d_cols:
        return df

    d = df[d_cols]
    df["D_sum"]     = d.sum(axis=1)
    df["D_abs_sum"] = d.abs().sum(axis=1)
    df["D_bull"]    = (d == 1).sum(axis=1)
    df["D_bear"]    = (d == -1).sum(axis=1)
    df["D_net"]     = df["D_bull"] - df["D_bear"]
    df["D_signal"]  = df["D_net"] / (df["D_abs_sum"] + EPS)  # normalized to [-1, 1]

    for pfx in ["E", "M", "P", "V", "S"]:
        mc = f"{pfx}_mean"
        if mc not in df.columns:
            continue
        df[f"{pfx}_x_regime"] = df[mc] * df["D_net"]
        df[f"{pfx}_x_bull"]   = df[mc] * df["D_bull"]
        df[f"{pfx}_x_bear"]   = df[mc] * df["D_bear"]
        df[f"{pfx}_bull_adj"] = df[mc] * (df["D_net"] > 0).astype(float)
        df[f"{pfx}_bear_adj"] = df[mc] * (df["D_net"] < 0).astype(float)

    print(f"  [5] Regime-conditioned → {df.shape[1]} cols")
    return df


# BLOCK 6 — CROSS-FAMILY INTERACTIONS
# Captures relationships between signal domains via product, ratio, and
# difference features. Each pair is economically motivated:
#   E×M  — economic data confirming or contradicting market signal
#   P×V  — price momentum adjusted by volatility regime
#   S×I  — sentiment and institutional flow agreement vs divergence
#   E×M×D — triple interaction conditioned on regime direction

def block_cross_family(df):
    df = df.copy()
    pairs = [
        ("E", "M"), ("P", "V"), ("S", "I"),
        ("M", "V"), ("E", "S"), ("P", "I"),
    ]
    for a, b in pairs:
        ca, cb = f"{a}_mean", f"{b}_mean"
        if ca not in df.columns or cb not in df.columns:
            continue
        df[f"{a}x{b}_prod"]    = df[ca] * df[cb]
        df[f"{a}x{b}_ratio"]   = df[ca] / (df[cb].abs() + EPS)
        df[f"{a}x{b}_diff"]    = df[ca] - df[cb]
        df[f"{a}x{b}_absdiff"] = (df[ca] - df[cb]).abs()

    if all(c in df.columns for c in ["E_mean", "M_mean", "D_net"]):
        df["EMD_triple"] = df["E_mean"] * df["M_mean"] * df["D_net"]

    # price and sentiment pointing in opposite directions signals potential reversal
    if "P_mean" in df.columns and "S_mean" in df.columns:
        df["PS_divergence"] = (np.sign(df["P_mean"]) != np.sign(df["S_mean"])).astype(float)

    print(f"  [6] Cross-family → {df.shape[1]} cols")
    return df


# BLOCK 7 — ROLLING AND LAG FEATURES
# Builds momentum and mean-reversion signals over windows of 3, 5, and 10
# periods. Applied to aggregate signals only to keep dimensionality manageable.
# mom{w} = current value minus rolling mean (deviation from recent trend).
# rz{w}  = that deviation normalized by rolling std (z-score in window).
# mom_cross = short MA minus long MA, a standard momentum crossover signal.
# Data is sorted by date_id before rolling to preserve temporal ordering.

ROLL_TARGETS = [
    "E_mean", "M_mean", "P_mean", "V_mean", "S_mean", "I_mean",
    "D_net", "D_signal", "IR_M", "IR_E", "momentum_sharpe",
    "ExM_prod", "PxV_prod",
]
WINDOWS = [3, 5, 10]

def block_rolling(df, is_train=True):
    df = df.copy()
    if "date_id" not in df.columns:
        return df

    df = df.sort_values("date_id").reset_index(drop=True)

    for col in ROLL_TARGETS:
        if col not in df.columns:
            continue
        s = df[col]
        for w in WINDOWS:
            rm = s.rolling(w, min_periods=1).mean()
            rs = s.rolling(w, min_periods=1).std().fillna(0)
            df[f"{col}_rmean{w}"] = rm
            df[f"{col}_rstd{w}"]  = rs
            df[f"{col}_mom{w}"]   = s - rm
            df[f"{col}_rz{w}"]    = (s - rm) / (rs + EPS)

        fill = s.mean()
        df[f"{col}_lag1"] = s.shift(1).fillna(fill)
        df[f"{col}_lag2"] = s.shift(2).fillna(fill)
        df[f"{col}_lag3"] = s.shift(3).fillna(fill)

        if f"{col}_rmean3" in df.columns and f"{col}_rmean10" in df.columns:
            df[f"{col}_mom_cross"] = df[f"{col}_rmean3"] - df[f"{col}_rmean10"]

    label = "train" if is_train else "test"
    print(f"  [7] Rolling ({label}) → {df.shape[1]} cols")
    return df


# BLOCK 8 — POLYNOMIAL FEATURES
# Squares and cubics applied selectively to the most informative signals.
# Cubic terms capture asymmetric response (e.g. large positive vs large
# negative signals behaving differently). Applied only to vol-adjusted
# and regime signals where non-linearity is most likely to be meaningful.

POLY_TARGETS = [
    "E_vol_adj", "M_vol_adj", "P_vol_adj",
    "IR_M", "IR_E", "momentum_sharpe",
    "D_signal", "EMD_triple",
]

def block_polynomial(df):
    df = df.copy()
    for col in POLY_TARGETS:
        if col not in df.columns:
            continue
        df[f"{col}_sq"]   = df[col] ** 2
        df[f"{col}_cb"]   = df[col] ** 3
        df[f"{col}_abs"]  = df[col].abs()
        df[f"{col}_sign"] = np.sign(df[col])
    print(f"  [8] Polynomial → {df.shape[1]} cols")
    return df


# BLOCK 9 — LIGHTGBM-BASED FEATURE SELECTION
# Uses a tree-based model to identify the most predictive features rather than
# relying on simple correlation. This captures nonlinear relationships and
# feature interactions, which are common in financial data. Instead of removing
# features based on weak individual correlation, we keep the top N features
# ranked by model importance. This dramatically reduces noise while preserving
# useful signal combinations.

def block_feature_selection(train, test, top_n=100):
    print("  [9] LightGBM feature selection")

    # Separate features and target
    X = train.drop(columns=[TARGET_COL])
    y = train[TARGET_COL]

    # Train a lightweight model for feature importance estimation
    model = lgb.LGBMRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )
    model.fit(X, y)

    # Rank features by importance
    importances = model.feature_importances_
    feat_names = X.columns

    imp_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    top_features = imp_df.head(top_n)["feature"].tolist()

    print(f"      Keeping top {top_n} features (model-based importance)")

    # Apply selection
    train = train[top_features + [TARGET_COL]]
    test  = test[[c for c in top_features if c in test.columns]]

    return train, test

# ALIGNMENT
# Ensures train and test share the same feature columns after all blocks.
# Any feature column present in train but missing from test is filled with 0.

def cleanup_and_align(train, test):
    train = train.fillna(0)
    test = test.fillna(0)
    for col in LEAKAGE:
        if col in train.columns: train = train.drop(columns=[col])
        if col in test.columns:  test  = test.drop(columns=[col])

    train_feat = [c for c in train.columns if c != TARGET_COL]
    for c in train_feat:
        if c not in test.columns:
            test[c] = 0.0

    extra = [c for c in test.columns if c not in train.columns]
    test  = test.drop(columns=extra)

    print(f"\n  Final → Train: {train.shape}  |  Test: {test.shape}")
    return train, test


# MAIN
# Stateless blocks (1-6, 8) are applied to both train and test independently.
# Block 7 (rolling) runs separately since it requires temporal ordering.
# Block 9 (selection) fits thresholds on train and applies the same drops to test.

def main():
    print("\n" + "█" * 70)
    print("  CS 6140 — Hull Tactical  |  Feature Engineering")
    print("█" * 70 + "\n")

    train, test = load_preprocessed()
    print("\n--- Building features ---")

    stateless_blocks = [
        block_group_stats,
        block_rank_features,
        block_zscore_features,
        block_vol_normalized,
        block_regime_conditioned,
        block_cross_family,
        block_polynomial,
    ]

    for fn in stateless_blocks:
        train = fn(train)
        test  = fn(test)

    train = block_rolling(train, is_train=True)
    test  = block_rolling(test,  is_train=False)

    train, test = block_feature_selection(train, test, top_n=100)
    train, test = cleanup_and_align(train, test)

    train.to_csv(os.path.join(OUTPUT_DIR, "train_features.csv"), index=False)
    test.to_csv(os.path.join(OUTPUT_DIR,  "test_features.csv"),  index=False)

    print(f"\n  Saved → outputs/train_features.csv")
    print(f"  Saved → outputs/test_features.csv")
    print("\n" + "█" * 70)
    print("  Feature engineering v2 complete.")
    print("█" * 70 + "\n")

    return train, test


if __name__ == "__main__":
    train_fe, test_fe = main()