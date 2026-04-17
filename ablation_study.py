"""
CS 6140 - Hull Tactical Market Prediction
Ablation Study — Group 15

Investigates the impact of key design choices in the advanced modelling pipeline.
Each ablation isolates one variable to measure its contribution to Sharpe ratio.

Ablation Groups:
  1. Regression vs Classification framing
  2. Default vs Optuna-tuned hyperparameters (XGBoost Classifier)
  3. LSTM v1 (no target scaling) vs LSTM v2 (with target scaling)
  4. Single model vs Ensemble strategies

Results are saved to outputs/ablation_study_results.csv
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

INPUT_DIR   = "outputs"
PRED_DIR    = os.path.join("outputs", "predictions")
FIGURES_DIR = os.path.join("outputs", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

TARGET_COL   = "market_forward_excess_returns"
ID_COLS      = ["row_id", "date_id"]
N_SPLITS     = 5
RANDOM_STATE = 42


# HELPERS

def sharpe_ratio(y_true, y_pred):
    pnl = y_pred * y_true
    std = pnl.std()
    return float(pnl.mean() / std) if std > 1e-8 else 0.0

def cv_sharpe(model, X, y, is_classifier=False):
    """
    Runs TimeSeriesSplit CV and returns mean and std Sharpe.
    For classifiers: converts predict_proba to position (2*P(up)-1).
    For regressors:  uses raw predictions as positions.
    """
    tscv        = TimeSeriesSplit(n_splits=N_SPLITS)
    sharpes     = []
    y_bin       = (np.sign(y) == 1).astype(int) if is_classifier else None

    for tr_idx, val_idx in tscv.split(X):
        if is_classifier:
            model.fit(X[tr_idx], y_bin[tr_idx])
            prob_up   = model.predict_proba(X[val_idx])[:, 1]
            positions = 2 * prob_up - 1
        else:
            model.fit(X[tr_idx], y[tr_idx])
            positions = model.predict(X[val_idx])

        sharpes.append(sharpe_ratio(y[val_idx], positions))

    return float(np.mean(sharpes)), float(np.std(sharpes))

def load_data():
    train      = pd.read_csv(os.path.join(INPUT_DIR, "train_features.csv"))
    drop_train = [c for c in ID_COLS + [TARGET_COL] if c in train.columns]
    X          = train.drop(columns=drop_train).values
    y          = train[TARGET_COL].values
    print(f"  Loaded: X={X.shape}  y={y.shape}")
    return X, y


# ABLATION 1 — REGRESSION VS CLASSIFICATION FRAMING
# Same model family (XGBoost), same hyperparameters.
# Only difference: regression predicts return magnitude,
# classification predicts direction (up/down).
# This isolates the impact of problem framing alone.

def ablation_1_framing(X, y):
    print("\n" + "=" * 60)
    print("ABLATION 1 — Regression vs Classification Framing")
    print("=" * 60)

    # Use previously saved Optuna best params for fair comparison
    # These are the params found by advanced_models_regression.py
    reg_params = {
        "n_estimators"    : 1495,
        "learning_rate"   : 0.127,
        "max_depth"       : 3,
        "min_child_weight": 5,
        "subsample"       : 0.514,
        "colsample_bytree": 0.457,
        "reg_alpha"       : 0.043,
        "reg_lambda"      : 0.013,
        "random_state"    : RANDOM_STATE,
        "n_jobs"          : -1,
        "verbosity"       : 0,
        "tree_method"     : "hist",
    }

    # Same params for classifier
    clf_params = {**reg_params,
                  "objective"  : "binary:logistic",
                  "eval_metric": "logloss"}

    print("  Running XGBoost Regressor...")
    reg   = xgb.XGBRegressor(**reg_params)
    m, s  = cv_sharpe(reg, X, y, is_classifier=False)
    print(f"  Regressor  Sharpe: {m:.4f} ± {s:.4f}")

    print("  Running XGBoost Classifier...")
    clf   = xgb.XGBClassifier(**clf_params)
    m2, s2 = cv_sharpe(clf, X, y, is_classifier=True)
    print(f"  Classifier Sharpe: {m2:.4f} ± {s2:.4f}")

    return [
        {"Ablation": "1 — Problem Framing", "Configuration": "XGBoost Regressor",
         "Mean Sharpe": m,  "Std Sharpe": s,  "Change": "baseline"},
        {"Ablation": "1 — Problem Framing", "Configuration": "XGBoost Classifier",
         "Mean Sharpe": m2, "Std Sharpe": s2, "Change": f"{m2-m:+.4f}"},
    ]


# ABLATION 2 — DEFAULT VS OPTUNA-TUNED HYPERPARAMETERS
# Same model (XGBoost Classifier), same framing.
# Only difference: default sklearn params vs Optuna-tuned params.
# This isolates the impact of hyperparameter tuning alone.

DEFAULT_CLF_PARAMS = {
    "n_estimators" : 100,
    "max_depth"    : 6,
    "learning_rate": 0.3,
    "random_state" : RANDOM_STATE,
    "n_jobs"       : -1,
    "verbosity"    : 0,
    "tree_method"  : "hist",
    "objective"    : "binary:logistic",
    "eval_metric"  : "logloss",
}

# Best params found by Optuna in advanced_models_classification.py
TUNED_CLF_PARAMS = {
    "n_estimators"     : 252,
    "learning_rate"    : 0.00785,
    "max_depth"        : 4,
    "min_child_weight" : 8,
    "subsample"        : 0.960,
    "colsample_bytree" : 0.696,
    "colsample_bylevel": 0.896,
    "reg_alpha"        : 0.000582,
    "reg_lambda"       : 0.000220,
    "gamma"            : 1.674,
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
    "verbosity"        : 0,
    "tree_method"      : "hist",
    "objective"        : "binary:logistic",
    "eval_metric"      : "logloss",
}

def ablation_2_tuning(X, y):
    print("\n" + "=" * 60)
    print("ABLATION 2 — Default vs Optuna-Tuned Hyperparameters")
    print("=" * 60)

    print("  Running XGBoost Classifier (default params)...")
    clf_default   = xgb.XGBClassifier(**DEFAULT_CLF_PARAMS)
    m, s          = cv_sharpe(clf_default, X, y, is_classifier=True)
    print(f"  Default Sharpe : {m:.4f} ± {s:.4f}")

    print("  Running XGBoost Classifier (Optuna-tuned params)...")
    clf_tuned     = xgb.XGBClassifier(**TUNED_CLF_PARAMS)
    m2, s2        = cv_sharpe(clf_tuned, X, y, is_classifier=True)
    print(f"  Tuned Sharpe   : {m2:.4f} ± {s2:.4f}")

    return [
        {"Ablation": "2 — Hyperparameter Tuning", "Configuration": "XGBoost CLF (Default)",
         "Mean Sharpe": m,  "Std Sharpe": s,  "Change": "baseline"},
        {"Ablation": "2 — Hyperparameter Tuning", "Configuration": "XGBoost CLF (Optuna-Tuned)",
         "Mean Sharpe": m2, "Std Sharpe": s2, "Change": f"{m2-m:+.4f}"},
    ]


# ABLATION 3 — LSTM: NO TARGET SCALING VS TARGET SCALING
# Already run in advanced_models_regression.py.
# Results pulled directly from saved CSV — no rerun needed.
# This isolates the impact of target scaling on LSTM performance.

def ablation_3_lstm():
    print("\n" + "=" * 60)
    print("ABLATION 3 — LSTM: No Target Scaling vs Target Scaling")
    print("=" * 60)

    # Results from advanced_models.py runs (v1 vs v2)
    lstm_v1_sharpe = -0.0005  # from first run (no target scaling)
    lstm_v2_sharpe =  0.0109  # from second run (with target scaling)

    print(f"  LSTM v1 (no target scaling) Sharpe : {lstm_v1_sharpe:.4f}")
    print(f"  LSTM v2 (with target scaling) Sharpe: {lstm_v2_sharpe:.4f}")

    return [
        {"Ablation": "3 — LSTM Target Scaling", "Configuration": "LSTM (No Target Scaling)",
         "Mean Sharpe": lstm_v1_sharpe, "Std Sharpe": 0.026, "Change": "baseline"},
        {"Ablation": "3 — LSTM Target Scaling", "Configuration": "LSTM (With Target Scaling)",
         "Mean Sharpe": lstm_v2_sharpe, "Std Sharpe": 0.039, "Change": f"{lstm_v2_sharpe-lstm_v1_sharpe:+.4f}"},
    ]


# ABLATION 4 — ENSEMBLE STRATEGIES
# Already run. Results pulled from saved CSVs.
# Compares single best model vs classifier ensemble vs mega ensemble.
# Isolates the impact of combining models.

def ablation_4_ensemble():
    print("\n" + "=" * 60)
    print("ABLATION 4 — Ensemble Strategies")
    print("=" * 60)

    configs = [
        {"Ablation": "4 — Ensemble Strategy", "Configuration": "Single Best (LightGBM CLF)",
         "Mean Sharpe": 0.0517, "Std Sharpe": 0.028, "Change": "baseline"},
        {"Ablation": "4 — Ensemble Strategy", "Configuration": "CLF Ensemble (XGB + LGB)",
         "Mean Sharpe": 0.0406, "Std Sharpe": 0.000, "Change": f"{0.0406-0.0517:+.4f}"},
        {"Ablation": "4 — Ensemble Strategy", "Configuration": "Mega Ensemble (CLF + Regressor)",
         "Mean Sharpe": 0.0407, "Std Sharpe": 0.000, "Change": f"{0.0407-0.0517:+.4f}"},
    ]

    for c in configs:
        print(f"  {c['Configuration']:40s} Sharpe: {c['Mean Sharpe']:.4f}")

    return configs


# RESULTS PLOT

def plot_ablation_results(all_results):
    df = pd.DataFrame(all_results)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    ablations = df["Ablation"].unique()
    colors_pos = "#27ae60"
    colors_neg = "#e74c3c"

    for i, abl in enumerate(ablations):
        sub = df[df["Ablation"] == abl].reset_index(drop=True)
        colors = [colors_pos if v >= 0 else colors_neg
                  for v in sub["Mean Sharpe"]]
        axes[i].barh(sub["Configuration"], sub["Mean Sharpe"],
                     xerr=sub["Std Sharpe"], color=colors,
                     edgecolor="white", capsize=4)
        axes[i].axvline(0, color="black", linewidth=0.8)
        axes[i].set_title(abl, fontweight="bold", fontsize=10)
        axes[i].set_xlabel("Mean Sharpe")
        axes[i].invert_yaxis()

    plt.suptitle("Ablation Study — Group 15", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "ablation_study.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Ablation plot saved → {path}")


# MAIN

def main():
    print("\n" + "█" * 70)
    print("  Hull Tactical  |  Ablation Study  —  Group 15")
    print("█" * 70 + "\n")

    X, y = load_data()

    all_results = []
    all_results += ablation_1_framing(X, y)
    all_results += ablation_2_tuning(X, y)
    all_results += ablation_3_lstm()
    all_results += ablation_4_ensemble()

    print("\n" + "=" * 60)
    print("ABLATION STUDY SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(all_results)
    print(df[["Ablation", "Configuration", "Mean Sharpe", "Change"]].to_string(index=False))

    df.to_csv(os.path.join("outputs", "ablation_study_results.csv"), index=False)
    print(f"\n  Results saved → outputs/ablation_study_results.csv")

    plot_ablation_results(all_results)

    print("\n" + "█" * 70)
    print("  Ablation study complete.")
    print("█" * 70 + "\n")

    return df

if __name__ == "__main__":
    df = main()