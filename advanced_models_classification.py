"""
CS 6140 - Hull Tactical Market Prediction
Advanced Models Script — Group 15

  1. XGBoost Classifier  — predicts market direction (+1/-1), uses
                           predictions as position sizes for Sharpe
  2. LightGBM Classifier — same approach, different base learner
  3. Classifier Ensemble — combines both classifiers via Sharpe-weighted blend
  4. Mega Ensemble       — best classifier + best regressor (XGBoost v2)

All models use TimeSeriesSplit(5) — no temporal leakage.
Metric: Sharpe Ratio = mean(y_pred * y_true) / std(y_pred * y_true)
        where y_pred ∈ {-1, +1} (direction) and y_true = actual returns
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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  [!] XGBoost not installed — run: pip install xgboost")

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
    """
    Core competition metric.
    y_pred is treated as a position size — PnL = y_pred * y_true.
    For classifiers: y_pred ∈ {-1, +1} gives maximum position sizing.
    For regressors:  y_pred is the raw predicted return.
    """
    pnl = y_pred * y_true
    std = pnl.std()
    return float(pnl.mean() / std) if std > 1e-8 else 0.0

def classification_metrics(y_true_returns, y_pred_direction):
    """
    Metrics for direction-prediction models.
    y_true_direction = sign of actual returns (ground truth direction).
    y_pred_direction = predicted direction {-1, +1}.
    Sharpe uses actual return magnitudes, not just direction.
    """
    y_true_dir  = np.sign(y_true_returns)
    y_pred_hard = np.sign(y_pred_direction)
    acc         = accuracy_score(y_true_dir, y_pred_hard)
    sharpe      = sharpe_ratio(y_true_returns, y_pred_direction)
    rmse        = float(np.sqrt(mean_squared_error(y_true_returns,
                  y_pred_direction * np.abs(y_true_returns).mean())))
    return {"Accuracy": acc, "Sharpe": sharpe, "RMSE": rmse}

def load_features():
    train = pd.read_csv(os.path.join(INPUT_DIR, "train_features.csv"))
    test  = pd.read_csv(os.path.join(INPUT_DIR, "test_features.csv"))
    print(f"  Train features : {train.shape}")
    print(f"  Test  features : {test.shape}")
    return train, test

def prepare_arrays(train, test):
    drop_train = [c for c in ID_COLS + [TARGET_COL] if c in train.columns]
    X_train    = train.drop(columns=drop_train).values
    y_train    = train[TARGET_COL].values
    feat_names = train.drop(columns=drop_train).columns.tolist()
    X_test     = test[[c for c in feat_names if c in test.columns]].values
    test_ids   = test["row_id"].values if "row_id" in test.columns else np.arange(len(test))

    # Binary direction labels for classifiers: +1 if return > 0, -1 otherwise
    y_direction = np.where(y_train > 0, 1, -1)

    print(f"  X_train    : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test     : {X_test.shape}")
    print(f"  Positive days : {(y_direction == 1).sum()}  "
          f"| Negative days : {(y_direction == -1).sum()}")
    return X_train, y_train, y_direction, X_test, test_ids, feat_names

def save_submission(test_ids, preds, model_name):
    df = pd.DataFrame({
        "row_id"                       : test_ids,
        "market_forward_excess_returns": preds,
    })
    path = os.path.join(PRED_DIR, f"submission_{model_name}.csv")
    df.to_csv(path, index=False)
    print(f"  Submission saved → {path}")

def plot_feature_importance(importances, feat_names, model_name, top_n=25):
    idx   = np.argsort(importances)[::-1][:top_n]
    vals  = importances[idx]
    names = [feat_names[i] for i in idx]
    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.35)))
    ax.barh(names[::-1], vals[::-1], color="#8e44ad", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"feature_importance_{model_name.lower().replace(' ','_')}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance saved → {path}")


# MODEL 1 — XGBOOST CLASSIFIER + OPTUNA
# Predicts market direction (+1/-1) using binary classification.
# predict_proba gives P(up) — we convert to position:
#   position = 2 * P(up) - 1  ∈ (-1, +1)
# This is smoother than hard {-1,+1} and encodes confidence.

def tune_xgb_clf_optuna(X_train, y_train, y_direction, n_trials=100):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 9),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 20),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.3, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "gamma"            : trial.suggest_float("gamma", 0.0, 2.0),
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
            "verbosity"        : 0,
            "tree_method"      : "hist",
            "objective"        : "binary:logistic",
            "eval_metric"      : "logloss",
        }

        sharpes = []
        # Map {-1,+1} → {0,1} for XGBoost binary classification
        y_bin = (y_direction == 1).astype(int)

        for tr_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train[tr_idx], y_bin[tr_idx],
                eval_set=[(X_train[val_idx], y_bin[val_idx])],
                verbose=False
            )
            # Probability → continuous position in (-1, +1)
            prob_up  = model.predict_proba(X_train[val_idx])[:, 1]
            position = 2 * prob_up - 1
            pnl      = position * y_train[val_idx]
            sharpe   = pnl.mean() / (pnl.std() + 1e-8)
            sharpes.append(sharpe)

        return float(np.mean(sharpes))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best Sharpe (Optuna): {study.best_value:.4f}")
    print(f"  Best params        : {study.best_params}")
    return study.best_params

def train_xgb_classifier(X_train, y_train, y_direction, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 1 — XGBoost Classifier + Optuna (100 trials)")
    print("=" * 60)

    if not HAS_XGB:
        print("  Skipping — XGBoost not installed.")
        return {"model": "XGBoost-CLF", "mean_Sharpe": 0, "std_Sharpe": 0,
                "oof_preds": np.zeros(len(y_train)), "test_preds": np.zeros(len(X_test))}

    print("  Running Optuna (100 trials)...")
    best_params = tune_xgb_clf_optuna(X_train, y_train, y_direction, n_trials=100)
    best_params.update({
        "random_state": RANDOM_STATE,
        "n_jobs"      : -1,
        "verbosity"   : 0,
        "tree_method" : "hist",
        "objective"   : "binary:logistic",
        "eval_metric" : "logloss",
    })

    y_bin        = (y_direction == 1).astype(int)
    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds    = np.zeros(len(y_train))
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m = xgb.XGBClassifier(**best_params)
        m.fit(X_train[tr_idx], y_bin[tr_idx], verbose=False)
        prob_up  = m.predict_proba(X_train[val_idx])[:, 1]
        position = 2 * prob_up - 1
        oof_preds[val_idx] = position
        metrics = classification_metrics(y_train[val_idx], position)
        metrics["fold"] = fold
        fold_results.append(metrics)

    df = pd.DataFrame(fold_results)
    summary = {
        "model"        : "XGBoost-CLF",
        "mean_Sharpe"  : df["Sharpe"].mean(),
        "std_Sharpe"   : df["Sharpe"].std(),
        "mean_Accuracy": df["Accuracy"].mean(),
        "mean_RMSE"    : df["RMSE"].mean(),
        "mean_R2"      : 0.0,
        "oof_preds"    : oof_preds,
    }

    print(f"\n  XGBoost Classifier")
    print(f"  Sharpe   : {summary['mean_Sharpe']:.4f} ± {summary['std_Sharpe']:.4f}")
    print(f"  Accuracy : {summary['mean_Accuracy']:.4f}")
    print(df[["fold","Sharpe","Accuracy"]].to_string(index=False))

    # Final fit
    final = xgb.XGBClassifier(**best_params)
    final.fit(X_train, y_bin, verbose=False)
    prob_up    = final.predict_proba(X_test)[:, 1]
    test_preds = 2 * prob_up - 1

    save_submission(test_ids, test_preds, "xgb_classifier")
    joblib.dump(final, os.path.join(MODEL_DIR, "xgb_classifier.pkl"))
    plot_feature_importance(final.feature_importances_, feat_names, "XGBoost-CLF")

    summary["test_preds"]  = test_preds
    summary["best_params"] = str(best_params)
    return summary


# MODEL 2 — LIGHTGBM CLASSIFIER + OPTUNA
# Same probability → position approach as XGBoost classifier.
# LightGBM tends to handle high-dimensional tabular data well and
# often finds different signal patterns than XGBoost.

def tune_lgb_clf_optuna(X_train, y_train, y_direction, n_trials=100):
    tscv  = TimeSeriesSplit(n_splits=N_SPLITS)
    y_bin = (y_direction == 1).astype(int)

    def objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 200, 1500),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.005, 0.15, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 9),
            "num_leaves"       : trial.suggest_int("num_leaves", 20, 150),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
            "verbosity"        : -1,
            "objective"        : "binary",
        }

        sharpes = []
        for tr_idx, val_idx in tscv.split(X_train):
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train[tr_idx], y_bin[tr_idx],
                eval_set=[(X_train[val_idx], y_bin[val_idx])],
                callbacks=[lgb.early_stopping(30, verbose=False),
                           lgb.log_evaluation(-1)]
            )
            prob_up  = model.predict_proba(X_train[val_idx])[:, 1]
            position = 2 * prob_up - 1
            pnl      = position * y_train[val_idx]
            sharpe   = pnl.mean() / (pnl.std() + 1e-8)
            sharpes.append(sharpe)

        return float(np.mean(sharpes))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=5)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"\n  Best Sharpe (Optuna): {study.best_value:.4f}")
    print(f"  Best params        : {study.best_params}")
    return study.best_params

def train_lgb_classifier(X_train, y_train, y_direction, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 2 — LightGBM Classifier + Optuna (100 trials)")
    print("=" * 60)

    if not HAS_LGB:
        print("  Skipping — LightGBM not installed.")
        return {"model": "LightGBM-CLF", "mean_Sharpe": 0, "std_Sharpe": 0,
                "oof_preds": np.zeros(len(y_train)), "test_preds": np.zeros(len(X_test))}

    print("  Running Optuna (100 trials)...")
    best_params = tune_lgb_clf_optuna(X_train, y_train, y_direction, n_trials=100)
    best_params.update({
        "random_state": RANDOM_STATE,
        "n_jobs"      : -1,
        "verbosity"   : -1,
        "objective"   : "binary",
    })

    y_bin        = (y_direction == 1).astype(int)
    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds    = np.zeros(len(y_train))
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m = lgb.LGBMClassifier(**best_params)
        m.fit(
            X_train[tr_idx], y_bin[tr_idx],
            eval_set=[(X_train[val_idx], y_bin[val_idx])],
            callbacks=[lgb.early_stopping(30, verbose=False),
                       lgb.log_evaluation(-1)]
        )
        prob_up  = m.predict_proba(X_train[val_idx])[:, 1]
        position = 2 * prob_up - 1
        oof_preds[val_idx] = position
        metrics = classification_metrics(y_train[val_idx], position)
        metrics["fold"] = fold
        fold_results.append(metrics)

    df = pd.DataFrame(fold_results)
    summary = {
        "model"        : "LightGBM-CLF",
        "mean_Sharpe"  : df["Sharpe"].mean(),
        "std_Sharpe"   : df["Sharpe"].std(),
        "mean_Accuracy": df["Accuracy"].mean(),
        "mean_RMSE"    : df["RMSE"].mean(),
        "mean_R2"      : 0.0,
        "oof_preds"    : oof_preds,
    }

    print(f"\n  LightGBM Classifier")
    print(f"  Sharpe   : {summary['mean_Sharpe']:.4f} ± {summary['std_Sharpe']:.4f}")
    print(f"  Accuracy : {summary['mean_Accuracy']:.4f}")
    print(df[["fold","Sharpe","Accuracy"]].to_string(index=False))

    final = lgb.LGBMClassifier(**best_params)
    final.fit(X_train, y_bin,
              callbacks=[lgb.log_evaluation(-1)])
    prob_up    = final.predict_proba(X_test)[:, 1]
    test_preds = 2 * prob_up - 1

    save_submission(test_ids, test_preds, "lgb_classifier")
    joblib.dump(final, os.path.join(MODEL_DIR, "lgb_classifier.pkl"))
    plot_feature_importance(final.feature_importances_, feat_names, "LightGBM-CLF")

    summary["test_preds"]  = test_preds
    summary["best_params"] = str(best_params)
    return summary


# MODEL 3 — CLASSIFIER ENSEMBLE
# Sharpe-weighted blend of XGBoost-CLF and LightGBM-CLF OOF predictions.

def classifier_ensemble(xgb_clf_summary, lgb_clf_summary, y_train, test_ids):
    print("\n" + "=" * 60)
    print("MODEL 3 — Classifier Ensemble (XGB-CLF + LGB-CLF)")
    print("=" * 60)

    w_xgb  = max(xgb_clf_summary["mean_Sharpe"], 0)
    w_lgb  = max(lgb_clf_summary["mean_Sharpe"], 0)
    total  = w_xgb + w_lgb

    if total < 1e-8:
        w_xgb, w_lgb = 0.5, 0.5
    else:
        w_xgb /= total
        w_lgb /= total

    print(f"  XGBoost-CLF weight  : {w_xgb:.3f}")
    print(f"  LightGBM-CLF weight : {w_lgb:.3f}")

    oof_blend  = w_xgb * xgb_clf_summary["oof_preds"] + w_lgb * lgb_clf_summary["oof_preds"]
    oof_sharpe = sharpe_ratio(y_train, oof_blend)
    print(f"  OOF Blend Sharpe    : {oof_sharpe:.4f}")

    test_blend = w_xgb * xgb_clf_summary["test_preds"] + w_lgb * lgb_clf_summary["test_preds"]
    save_submission(test_ids, test_blend, "clf_ensemble")

    return {
        "model"      : "CLF Ensemble",
        "mean_Sharpe": oof_sharpe,
        "std_Sharpe" : 0.0,
        "mean_R2"    : 0.0,
        "oof_preds"  : oof_blend,
        "test_preds" : test_blend,
    }


# MODEL 4 — MEGA ENSEMBLE
# Combines the best classifier ensemble with the best regressor (XGBoost v2).
# Load XGBoost v2 OOF predictions from disk if available,
# otherwise recompute them.

def mega_ensemble(clf_ens_summary, y_train, test_ids, X_train, X_test):
    print("\n" + "=" * 60)
    print("MODEL 4 — Mega Ensemble (CLF Ensemble + XGBoost Regressor)")
    print("=" * 60)

    # Try to load saved XGBoost v2 regressor
    xgb_reg_path = os.path.join(MODEL_DIR, "xgboost_v2.pkl")
    xgb_reg_sub  = os.path.join(PRED_DIR, "submission_xgboost_v2.csv")

    if os.path.exists(xgb_reg_path) and os.path.exists(xgb_reg_sub):
        print("  Loading saved XGBoost v2 regressor...")
        xgb_reg = joblib.load(xgb_reg_path)

        # Recompute OOF for the regressor
        tscv      = TimeSeriesSplit(n_splits=N_SPLITS)
        oof_reg   = np.zeros(len(y_train))
        for tr_idx, val_idx in tscv.split(X_train):
            xgb_reg.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
            oof_reg[val_idx] = xgb_reg.predict(X_train[val_idx])

        xgb_reg.fit(X_train, y_train, verbose=False)
        test_reg = xgb_reg.predict(X_test)

        reg_sharpe = sharpe_ratio(y_train, oof_reg)
        print(f"  XGBoost Regressor OOF Sharpe : {reg_sharpe:.4f}")
    else:
        print("  XGBoost v2 model not found — skipping regressor component.")
        print("  Run advanced_models_v2.py first to generate it.")
        return None

    # Sharpe-weighted blend
    w_clf = max(clf_ens_summary["mean_Sharpe"], 0)
    w_reg = max(reg_sharpe, 0)
    total = w_clf + w_reg

    if total < 1e-8:
        w_clf, w_reg = 0.5, 0.5
    else:
        w_clf /= total
        w_reg /= total

    print(f"  CLF Ensemble weight  : {w_clf:.3f}")
    print(f"  XGB Regressor weight : {w_reg:.3f}")

    oof_mega   = w_clf * clf_ens_summary["oof_preds"] + w_reg * oof_reg
    mega_sharpe = sharpe_ratio(y_train, oof_mega)
    print(f"  Mega OOF Sharpe      : {mega_sharpe:.4f}")

    test_mega = w_clf * clf_ens_summary["test_preds"] + w_reg * test_reg
    save_submission(test_ids, test_mega, "mega_ensemble")

    return {
        "model"      : "Mega Ensemble",
        "mean_Sharpe": mega_sharpe,
        "std_Sharpe" : 0.0,
        "mean_R2"    : 0.0,
    }


# COMPARISON PLOT

def plot_all_results(results):
    df = pd.DataFrame([r for r in results if r is not None])
    df = df[["model","mean_Sharpe","std_Sharpe"]].sort_values("mean_Sharpe", ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#27ae60" if v > 0 else "#e74c3c" for v in df["mean_Sharpe"]]
    ax.barh(df["model"], df["mean_Sharpe"],
            xerr=df["std_Sharpe"], color=colors, edgecolor="white", capsize=5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("All Models v3 — Sharpe Ratio Comparison", fontweight="bold", fontsize=13)
    ax.set_xlabel("Mean Sharpe Ratio")
    ax.invert_yaxis()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "all_models_v3_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved → {path}")


# MAIN

def main():
    print("\n" + "█" * 70)
    print("  Hull Tactical  |  Advanced Models v3  (Snehita Kandula)")
    print("█" * 70 + "\n")

    train, test = load_features()
    X_train, y_train, y_direction, X_test, test_ids, feat_names = prepare_arrays(train, test)

    # Model 1 — XGBoost Classifier
    xgb_clf_summary = train_xgb_classifier(
        X_train, y_train, y_direction, X_test, test_ids, feat_names)

    # Model 2 — LightGBM Classifier
    lgb_clf_summary = train_lgb_classifier(
        X_train, y_train, y_direction, X_test, test_ids, feat_names)

    # Model 3 — Classifier Ensemble
    clf_ens_summary = classifier_ensemble(
        xgb_clf_summary, lgb_clf_summary, y_train, test_ids)

    # Model 4 — Mega Ensemble (CLF + best regressor from v2)
    mega_summary = mega_ensemble(
        clf_ens_summary, y_train, test_ids, X_train, X_test)

    results = [xgb_clf_summary, lgb_clf_summary, clf_ens_summary, mega_summary]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — ADVANCED MODELS v3")
    print("=" * 60)
    df_results = pd.DataFrame([r for r in results if r is not None])
    print(df_results[["model","mean_Sharpe","std_Sharpe"]].to_string(index=False))

    df_results.to_csv(os.path.join("outputs", "advanced_model_results_v3.csv"), index=False)
    print(f"\n  Summary saved → outputs/advanced_model_results_v3.csv")

    plot_all_results(results)

    print("\n" + "█" * 70)
    print("  Advanced modelling v3 complete.")
    print("█" * 70 + "\n")

    return df_results

if __name__ == "__main__":
    df_results = main()