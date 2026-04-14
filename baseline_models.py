"""
CS 6140 - Hull Tactical Market Prediction
Baseline Models Script — v2  (Aasav Suthar)

Trains four models on the engineered feature set, evaluates each using
TimeSeriesSplit cross-validation, and outputs Kaggle-ready submission files.
LightGBM is tuned via Optuna; remaining models use fixed or CV-selected params.

Input  : outputs/train_features.csv, outputs/test_features.csv
Output : outputs/predictions/submission_<model>.csv
         outputs/model_results_summary_v2.csv
         outputs/figures/model_comparison_v2.png
         outputs/figures/feature_importance_<model>.png

Models : Ridge, Gradient Boosting, LightGBM (+ Optuna), Random Forest
Metric : Sharpe Ratio via TimeSeriesSplit(5) — no temporal data leakage
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.linear_model import Ridge, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  [!] LightGBM not installed — run: pip install lightgbm")

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("  [!] Optuna not installed — run: pip install optuna")

INPUT_DIR   = "outputs"
PRED_DIR    = os.path.join("outputs", "predictions")
MODEL_DIR   = os.path.join("outputs", "models")
FIGURES_DIR = os.path.join("outputs", "figures")

for d in [PRED_DIR, MODEL_DIR, FIGURES_DIR]:
    os.makedirs(d, exist_ok=True)

TARGET_COL   = "market_forward_excess_returns"
ID_COLS      = ["row_id", "date_id"]
N_SPLITS     = 5
RANDOM_STATE = 42


# HELPERS

def sharpe_ratio(y_true, y_pred):
    # competition metric: predictions treated as position sizes, PnL = pred × actual
    pnl = y_pred * y_true
    std = pnl.std()
    return float(pnl.mean() / std) if std > 1e-8 else 0.0

def regression_metrics(y_true, y_pred):
    return {
        "RMSE"  : float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE"   : float(mean_absolute_error(y_true, y_pred)),
        "R2"    : float(r2_score(y_true, y_pred)),
        "Sharpe": sharpe_ratio(y_true, y_pred),
    }

def load_features():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train_features.csv"))
    test  = pd.read_csv(os.path.join(INPUT_DIR, "test_features.csv"))
    print(f"  Train features : {train.shape}")
    print(f"  Test  features : {test.shape}")
    return train, test

def prepare_arrays(train, test):
    # X_test is aligned to train's exact feature columns to prevent shape mismatch
    drop_train      = [c for c in ID_COLS + [TARGET_COL] if c in train.columns]
    X_train         = train.drop(columns=drop_train).values
    y_train         = train[TARGET_COL].values
    feat_names      = train.drop(columns=drop_train).columns.tolist()
    X_test          = test[[c for c in feat_names if c in test.columns]].values
    test_ids        = test["row_id"].values if "row_id" in test.columns else np.arange(len(test))

    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    return X_train, y_train, X_test, test_ids, feat_names

def cv_evaluate(model, X, y, model_name):
    # TimeSeriesSplit preserves temporal order — validation always follows training in time
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X), 1):
        model.fit(X[tr_idx], y[tr_idx])
        y_pred = model.predict(X[val_idx])
        m = regression_metrics(y[val_idx], y_pred)
        m["fold"] = fold
        fold_results.append(m)

    df = pd.DataFrame(fold_results)
    summary = {
        "model"      : model_name,
        "mean_Sharpe": df["Sharpe"].mean(),
        "std_Sharpe" : df["Sharpe"].std(),
        "mean_RMSE"  : df["RMSE"].mean(),
        "mean_R2"    : df["R2"].mean(),
    }

    print(f"\n  {model_name}")
    print(f"  Sharpe : {summary['mean_Sharpe']:.4f} ± {summary['std_Sharpe']:.4f}")
    print(f"  RMSE   : {summary['mean_RMSE']:.6f}  |  R² : {summary['mean_R2']:.4f}")
    print(df[["fold","Sharpe","RMSE","R2"]].to_string(index=False))
    return summary

def save_submission(test_ids, preds, model_name):
    df = pd.DataFrame({
        "row_id"                       : test_ids,
        "market_forward_excess_returns": preds,
    })
    path = os.path.join(PRED_DIR, f"submission_{model_name}.csv")
    df.to_csv(path, index=False)
    print(f"  Submission saved → {path}")


# MODEL 1 — RIDGE REGRESSION
# Serves as a regularized linear baseline. With 300+ features, a high alpha
# is necessary to prevent numerical instability. RidgeCV selects the optimal
# alpha via internal cross-validation over a log-spaced grid.

def train_ridge(X_train, y_train, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 1 — Ridge Regression")
    print("=" * 60)

    alphas   = np.logspace(2, 7, 100)
    ridge_cv = RidgeCV(alphas=alphas, cv=TimeSeriesSplit(n_splits=N_SPLITS))
    ridge_cv.fit(X_train, y_train)
    best_alpha = ridge_cv.alpha_
    print(f"  Best alpha: {best_alpha:.4f}")

    model   = Ridge(alpha=best_alpha)
    summary = cv_evaluate(model, X_train, y_train, "Ridge")
    model.fit(X_train, y_train)
    save_submission(test_ids, model.predict(X_test), "ridge")
    joblib.dump(model, os.path.join(MODEL_DIR, "ridge.pkl"))

    coef_df = pd.DataFrame({"feature": feat_names, "coef": model.coef_})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    top = coef_df.nlargest(20, "abs_coef")
    plot_feature_importance(top["abs_coef"].values, top["feature"].tolist(), "Ridge_coef")

    summary["best_alpha"] = best_alpha
    return summary


# MODEL 2 — GRADIENT BOOSTING REGRESSOR
# Non-linear baseline using sklearn's GBM — no additional dependencies required.
# Shallow trees with subsampling provide sufficient regularization for this
# dataset size (~9k rows). Learning rate is kept low for stable convergence.

def train_gradient_boosting(X_train, y_train, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 2 — Gradient Boosting Regressor")
    print("=" * 60)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_leaf=20,
        subsample=0.8,          # row subsampling for regularization
        max_features="sqrt",    # feature subsampling per tree
        random_state=RANDOM_STATE,
    )
    summary = cv_evaluate(model, X_train, y_train, "GradientBoosting")
    model.fit(X_train, y_train)
    save_submission(test_ids, model.predict(X_test), "gradient_boosting")
    joblib.dump(model, os.path.join(MODEL_DIR, "gradient_boosting.pkl"))
    plot_feature_importance(model.feature_importances_, feat_names, "GradientBoosting")
    return summary


# MODEL 3 — LIGHTGBM + OPTUNA TUNING
# LightGBM handles high-dimensional tabular data efficiently and natively
# supports missing values. Optuna uses TPE (Tree-structured Parzen Estimator)
# to search the hyperparameter space — more sample-efficient than grid search.
# The objective maximizes mean Sharpe across TimeSeriesSplit folds.
# Falls back to fixed defaults if Optuna is not installed.

def tune_lgbm_optuna(X_train, y_train, n_trials=30):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 100, 800),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 8),
            "num_leaves"       : trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
            "verbosity"        : -1,
        }
        sharpes = []
        for tr_idx, val_idx in tscv.split(X_train):
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
            )
            pnl = m.predict(X_train[val_idx]) * y_train[val_idx]
            sharpes.append(float(pnl.mean() / (pnl.std() + 1e-8)))
        return float(np.mean(sharpes))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best Sharpe (Optuna): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


def train_lightgbm(X_train, y_train, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 3 — LightGBM" + (" + Optuna Tuning" if HAS_OPTUNA else " (default params)"))
    print("=" * 60)

    if not HAS_LGB:
        print("  Skipping — LightGBM not installed.")
        return {"model": "LightGBM", "mean_Sharpe": 0, "std_Sharpe": 0, "mean_RMSE": 0, "mean_R2": 0}

    if HAS_OPTUNA:
        print("  Running Optuna (30 trials)...")
        best_params = tune_lgbm_optuna(X_train, y_train, n_trials=30)
        best_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    else:
        best_params = {
            "n_estimators": 500, "learning_rate": 0.05, "max_depth": 5,
            "num_leaves": 63, "min_child_samples": 20, "subsample": 0.8,
            "colsample_bytree": 0.7, "reg_alpha": 0.1, "reg_lambda": 1.0,
            "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
        }

    model   = lgb.LGBMRegressor(**best_params)
    summary = cv_evaluate(model, X_train, y_train, "LightGBM")
    model.fit(X_train, y_train)
    save_submission(test_ids, model.predict(X_test), "lightgbm")
    joblib.dump(model, os.path.join(MODEL_DIR, "lightgbm.pkl"))
    plot_feature_importance(model.feature_importances_, feat_names, "LightGBM")

    summary["best_params"] = str(best_params)
    return summary


# MODEL 4 — RANDOM FOREST
# Fixed hyperparameters rather than a tuning run — RandomizedSearch on this
# feature count is slow and marginal gains over conservative defaults are small.
# max_depth=8 and min_samples_leaf=10 prevent overfitting at ~9k training rows.

def train_random_forest(X_train, y_train, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 4 — Random Forest")
    print("=" * 60)

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        max_features="sqrt",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    summary = cv_evaluate(model, X_train, y_train, "RandomForest")
    model.fit(X_train, y_train)
    save_submission(test_ids, model.predict(X_test), "random_forest")
    joblib.dump(model, os.path.join(MODEL_DIR, "random_forest.pkl"))
    plot_feature_importance(model.feature_importances_, feat_names, "RandomForest")
    return summary


# VISUALIZATIONS

def plot_feature_importance(importances, feat_names, model_name, top_n=25):
    idx   = np.argsort(importances)[::-1][:top_n]
    vals  = importances[idx]
    names = [feat_names[i] for i in idx]

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(names[::-1], vals[::-1], color="#9b59b6", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"feature_importance_{model_name.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance saved → {path}")

def plot_model_comparison(results):
    # three-panel comparison: Sharpe (with std error bars), RMSE, and R²
    df = pd.DataFrame(results)[["model","mean_Sharpe","std_Sharpe","mean_RMSE","mean_R2"]]
    df = df.sort_values("mean_Sharpe", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].barh(df["model"], df["mean_Sharpe"],
                 xerr=df["std_Sharpe"], color="#3498db", edgecolor="white", capsize=5)
    axes[0].set_title("Mean Sharpe (↑ better)", fontweight="bold")
    axes[0].axvline(0, color="black", linewidth=0.8)

    axes[1].barh(df["model"], df["mean_RMSE"], color="#e74c3c", edgecolor="white")
    axes[1].set_title("Mean RMSE (↓ better)", fontweight="bold")

    axes[2].barh(df["model"], df["mean_R2"], color="#2ecc71", edgecolor="white")
    axes[2].set_title("Mean R² (↑ better)", fontweight="bold")
    axes[2].axvline(0, color="black", linewidth=0.8)

    for ax in axes:
        ax.invert_yaxis()

    plt.suptitle("Model Comparison v2 — TimeSeriesSplit CV", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "model_comparison_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved → {path}")


# MAIN

def main():
    print("\n" + "█" * 70)
    print("  CS 6140 — Hull Tactical  |  Baseline Models v2  (Aasav Suthar)")
    print("█" * 70 + "\n")

    train, test = load_features()
    X_train, y_train, X_test, test_ids, feat_names = prepare_arrays(train, test)

    results = []
    results.append(train_ridge(X_train, y_train, X_test, test_ids, feat_names))
    results.append(train_gradient_boosting(X_train, y_train, X_test, test_ids, feat_names))
    results.append(train_lightgbm(X_train, y_train, X_test, test_ids, feat_names))
    results.append(train_random_forest(X_train, y_train, X_test, test_ids, feat_names))

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    df_results = pd.DataFrame(results)
    print(df_results[["model","mean_Sharpe","std_Sharpe","mean_RMSE","mean_R2"]].to_string(index=False))

    df_results.to_csv(os.path.join("outputs", "model_results_summary_v2.csv"), index=False)
    print(f"\n  Summary saved → outputs/model_results_summary_v2.csv")

    plot_model_comparison(results)

    best = max(results, key=lambda r: r["mean_Sharpe"])
    print(f"\n  Best model: {best['model']}  (Sharpe = {best['mean_Sharpe']:.4f})")
    print("  → Pass best submission CSV + .pkl to Snehita for ensemble.\n")

    print("█" * 70)
    print("  Baseline modelling v2 complete.")
    print("█" * 70 + "\n")

    return df_results


if __name__ == "__main__":
    df_results = main()