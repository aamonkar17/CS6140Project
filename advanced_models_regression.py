"""
CS 6140 - Hull Tactical Market Prediction
Advanced Models Script — v2

Models:
  1. XGBoost + Optuna (100 trials, expanded search space)
  2. LSTM (redesigned: target scaling, simpler arch, shorter seq, gradient clipping)
  3. Hybrid Ensemble (XGBoost + LSTM weighted by fold Sharpe)

Metric: Sharpe Ratio via TimeSeriesSplit(5)
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("  [!] TensorFlow not installed — run: pip install tensorflow")

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
SEQUENCE_LEN = 5


# HELPERS

def sharpe_ratio(y_true, y_pred):
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
    drop_train = [c for c in ID_COLS + [TARGET_COL] if c in train.columns]
    X_train    = train.drop(columns=drop_train).values
    y_train    = train[TARGET_COL].values
    feat_names = train.drop(columns=drop_train).columns.tolist()
    X_test     = test[[c for c in feat_names if c in test.columns]].values
    test_ids   = test["row_id"].values if "row_id" in test.columns else np.arange(len(test))
    print(f"  X_train : {X_train.shape}  |  y_train : {y_train.shape}")
    print(f"  X_test  : {X_test.shape}")
    return X_train, y_train, X_test, test_ids, feat_names

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
    ax.barh(names[::-1], vals[::-1], color="#e67e22", edgecolor="white")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, f"feature_importance_{model_name.lower()}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Feature importance saved → {path}")


# MODEL 1 — XGBOOST + OPTUNA (100 trials, expanded search space)

def tune_xgb_optuna(X_train, y_train, n_trials=100):
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
            "max_delta_step"   : trial.suggest_int("max_delta_step", 0, 5),
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
            "verbosity"        : 0,
            "tree_method"      : "hist",
        }

        sharpes = []
        for tr_idx, val_idx in tscv.split(X_train):
            model = xgb.XGBRegressor(**params)
            model.fit(
                X_train[tr_idx], y_train[tr_idx],
                eval_set=[(X_train[val_idx], y_train[val_idx])],
                verbose=False
            )
            preds  = model.predict(X_train[val_idx])
            pnl    = preds * y_train[val_idx]
            sharpe = pnl.mean() / (pnl.std() + 1e-8)
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

def train_xgboost(X_train, y_train, X_test, test_ids, feat_names):
    print("\n" + "=" * 60)
    print("MODEL 1 — XGBoost + Optuna (100 trials)")
    print("=" * 60)

    if not HAS_XGB:
        print("  Skipping — XGBoost not installed.")
        return {"model": "XGBoost", "mean_Sharpe": 0, "std_Sharpe": 0,
                "mean_RMSE": 0, "mean_R2": 0,
                "oof_preds": np.zeros(len(y_train)), "test_preds": np.zeros(len(X_test))}

    print("  Running Optuna (100 trials)...")
    best_params = tune_xgb_optuna(X_train, y_train, n_trials=100)
    best_params.update({
        "random_state": RANDOM_STATE,
        "n_jobs"      : -1,
        "verbosity"   : 0,
        "tree_method" : "hist",
    })

    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_preds    = np.zeros(len(y_train))
    fold_results = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        m = xgb.XGBRegressor(**best_params)
        m.fit(X_train[tr_idx], y_train[tr_idx], verbose=False)
        preds = m.predict(X_train[val_idx])
        oof_preds[val_idx] = preds
        metrics = regression_metrics(y_train[val_idx], preds)
        metrics["fold"] = fold
        fold_results.append(metrics)

    df = pd.DataFrame(fold_results)
    summary = {
        "model"      : "XGBoost",
        "mean_Sharpe": df["Sharpe"].mean(),
        "std_Sharpe" : df["Sharpe"].std(),
        "mean_RMSE"  : df["RMSE"].mean(),
        "mean_R2"    : df["R2"].mean(),
        "oof_preds"  : oof_preds,
    }

    print(f"\n  XGBoost")
    print(f"  Sharpe : {summary['mean_Sharpe']:.4f} ± {summary['std_Sharpe']:.4f}")
    print(f"  RMSE   : {summary['mean_RMSE']:.6f}  |  R² : {summary['mean_R2']:.4f}")
    print(df[["fold","Sharpe","RMSE","R2"]].to_string(index=False))

    final = xgb.XGBRegressor(**best_params)
    final.fit(X_train, y_train, verbose=False)
    test_preds = final.predict(X_test)

    save_submission(test_ids, test_preds, "xgboost_v2")
    joblib.dump(final, os.path.join(MODEL_DIR, "xgboost_v2.pkl"))
    plot_feature_importance(final.feature_importances_, feat_names, "XGBoost_v2")

    summary["test_preds"]  = test_preds
    summary["best_params"] = str(best_params)
    return summary


# MODEL 2 — LSTM (REDESIGNED)
# Key fixes vs v1:
#   1. Target scaled with StandardScaler then inverse-transformed — fixes huge RMSE
#   2. Simpler single LSTM layer — less overfitting on small dataset
#   3. Shorter sequence (5) — more usable training samples per fold
#   4. Huber loss — robust to outliers common in financial returns
#   5. Gradient clipping (clipnorm=1.0) — prevents exploding gradients

def plot_lstm_loss_curves(fold_histories):
    """
    Plots train vs validation loss curves for each CV fold.
    Useful for bias-variance analysis — diverging curves indicate overfitting,
    both curves high indicate underfitting.
    """
    fig, axes = plt.subplots(1, N_SPLITS, figsize=(18, 4))
    for i, hist in enumerate(fold_histories):
        axes[i].plot(hist["loss"],     label="Train Loss", color="#3498db")
        axes[i].plot(hist["val_loss"], label="Val Loss",   color="#e74c3c")
        axes[i].set_title(f"Fold {i+1}", fontweight="bold")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Huber Loss")
        axes[i].legend(fontsize=8)
    plt.suptitle("LSTM Training vs Validation Loss — All Folds",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "lstm_loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  LSTM loss curves saved → {path}")

def build_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X)):
        if i < seq_len:
            pad = np.zeros((seq_len - i - 1, X.shape[1]))
            seq = np.vstack([pad, X[:i+1]])
        else:
            seq = X[i-seq_len+1:i+1]
        Xs.append(seq)
        ys.append(y[i])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def build_lstm_model(seq_len, n_features):
    model = Sequential([
        LSTM(32, input_shape=(seq_len, n_features), return_sequences=False),
        BatchNormalization(),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dropout(0.1),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=5e-4, clipnorm=1.0),
        loss="huber"
    )
    return model

def train_lstm(X_train, y_train, X_test, test_ids):
    print("\n" + "=" * 60)
    print("MODEL 2 — LSTM (Redesigned v2)")
    print("=" * 60)

    if not HAS_TF:
        print("  Skipping — TensorFlow not installed.")
        return {"model": "LSTM", "mean_Sharpe": 0, "std_Sharpe": 0,
                "mean_RMSE": 0, "mean_R2": 0,
                "oof_preds": np.zeros(len(y_train)), "test_preds": np.zeros(len(X_test))}

    feat_scaler = StandardScaler()
    X_sc        = feat_scaler.fit_transform(X_train)
    X_te_sc     = feat_scaler.transform(X_test)

    tgt_scaler  = StandardScaler()
    y_sc        = tgt_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    X_seq,    y_seq = build_sequences(X_sc,    y_sc,                  SEQUENCE_LEN)
    X_te_seq, _     = build_sequences(X_te_sc, np.zeros(len(X_test)), SEQUENCE_LEN)

    tscv         = TimeSeriesSplit(n_splits=N_SPLITS)
    fold_results = []
    oof_preds    = np.zeros(len(y_train))
    fold_histories = []  # store loss curves for plotting

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0),
    ]

    print(f"  Sequence length : {SEQUENCE_LEN}")
    print(f"  Input shape     : ({SEQUENCE_LEN}, {X_train.shape[1]})\n")

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_seq), 1):
        print(f"  Fold {fold}/{N_SPLITS} ...", end=" ", flush=True)

        model = build_lstm_model(SEQUENCE_LEN, X_train.shape[1])
        history = model.fit(
            X_seq[tr_idx], y_seq[tr_idx],
            validation_data=(X_seq[val_idx], y_seq[val_idx]),
            epochs=150,
            batch_size=32,
            callbacks=callbacks,
            verbose=0
        )
        fold_histories.append(history.history)  # save loss history

        preds_sc = model.predict(X_seq[val_idx], verbose=0).flatten()
        preds    = tgt_scaler.inverse_transform(preds_sc.reshape(-1, 1)).flatten()
        oof_preds[val_idx] = preds

        m = regression_metrics(y_train[val_idx], preds)
        m["fold"] = fold
        fold_results.append(m)
        print(f"Sharpe={m['Sharpe']:.4f}  RMSE={m['RMSE']:.6f}")

    # Plot loss curves for all folds
    plot_lstm_loss_curves(fold_histories)

    df = pd.DataFrame(fold_results)
    summary = {
        "model"      : "LSTM",
        "mean_Sharpe": df["Sharpe"].mean(),
        "std_Sharpe" : df["Sharpe"].std(),
        "mean_RMSE"  : df["RMSE"].mean(),
        "mean_R2"    : df["R2"].mean(),
        "oof_preds"  : oof_preds,
    }

    print(f"\n  LSTM")
    print(f"  Sharpe : {summary['mean_Sharpe']:.4f} ± {summary['std_Sharpe']:.4f}")
    print(f"  RMSE   : {summary['mean_RMSE']:.6f}  |  R² : {summary['mean_R2']:.4f}")
    print(df[["fold","Sharpe","RMSE","R2"]].to_string(index=False))

    print("\n  Retraining on full data for test predictions...")
    final_model = build_lstm_model(SEQUENCE_LEN, X_train.shape[1])
    final_model.fit(X_seq, y_seq, epochs=80, batch_size=32, verbose=0)

    test_preds_sc = final_model.predict(X_te_seq, verbose=0).flatten()
    test_preds    = tgt_scaler.inverse_transform(
                        test_preds_sc.reshape(-1, 1)).flatten()

    save_submission(test_ids, test_preds, "lstm_v2")
    final_model.save(os.path.join(MODEL_DIR, "lstm_v2.keras"))
    print(f"  Model saved → {os.path.join(MODEL_DIR, 'lstm_v2.keras')}")

    summary["test_preds"] = test_preds
    return summary


# MODEL 3 — HYBRID ENSEMBLE
# Weights each model by its mean OOF Sharpe (floored at 0).
# Falls back to equal weighting if both Sharpes are negative.

def hybrid_ensemble(xgb_summary, lstm_summary, y_train, test_ids):
    print("\n" + "=" * 60)
    print("MODEL 3 — Hybrid Ensemble (XGBoost + LSTM)")
    print("=" * 60)

    xgb_sharpe  = max(xgb_summary["mean_Sharpe"], 0)
    lstm_sharpe = max(lstm_summary["mean_Sharpe"], 0)
    total       = xgb_sharpe + lstm_sharpe

    if total < 1e-8:
        w_xgb, w_lstm = 0.5, 0.5
    else:
        w_xgb  = xgb_sharpe  / total
        w_lstm = lstm_sharpe / total

    print(f"  XGBoost weight : {w_xgb:.3f}")
    print(f"  LSTM weight    : {w_lstm:.3f}")

    oof_blend  = (w_xgb * xgb_summary["oof_preds"] +
                  w_lstm * lstm_summary["oof_preds"])
    oof_sharpe = sharpe_ratio(y_train, oof_blend)
    print(f"  OOF Blend Sharpe : {oof_sharpe:.4f}")

    test_blend = (w_xgb * xgb_summary["test_preds"] +
                  w_lstm * lstm_summary["test_preds"])
    save_submission(test_ids, test_blend, "hybrid_ensemble")

    return {
        "model"      : "Hybrid Ensemble",
        "mean_Sharpe": oof_sharpe,
        "std_Sharpe" : 0.0,
        "mean_RMSE"  : 0.0,
        "mean_R2"    : 0.0,
    }


# COMPARISON PLOT

def plot_model_comparison(results):
    df = pd.DataFrame(results)[["model","mean_Sharpe","std_Sharpe","mean_RMSE","mean_R2"]]
    df = df.sort_values("mean_Sharpe", ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].barh(df["model"], df["mean_Sharpe"],
                 xerr=df["std_Sharpe"], color="#e67e22", edgecolor="white", capsize=5)
    axes[0].set_title("Mean Sharpe (↑ better)", fontweight="bold")
    axes[0].axvline(0, color="black", linewidth=0.8)

    axes[1].barh(df["model"], df["mean_RMSE"], color="#e74c3c", edgecolor="white")
    axes[1].set_title("Mean RMSE (↓ better)", fontweight="bold")

    axes[2].barh(df["model"], df["mean_R2"], color="#2ecc71", edgecolor="white")
    axes[2].set_title("Mean R² (↑ better)", fontweight="bold")
    axes[2].axvline(0, color="black", linewidth=0.8)

    for ax in axes:
        ax.invert_yaxis()

    plt.suptitle("Advanced Models v2 — TimeSeriesSplit CV", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "advanced_model_comparison_v2.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Comparison plot saved → {path}")


# MAIN

def main():
    print("\n" + "█" * 70)
    print("  Hull Tactical  |  Advanced Models v2  (Snehita Kandula)")
    print("█" * 70 + "\n")

    train, test = load_features()
    X_train, y_train, X_test, test_ids, feat_names = prepare_arrays(train, test)

    xgb_summary  = train_xgboost(X_train, y_train, X_test, test_ids, feat_names)
    lstm_summary = train_lstm(X_train, y_train, X_test, test_ids)
    ens_summary  = hybrid_ensemble(xgb_summary, lstm_summary, y_train, test_ids)

    results = [xgb_summary, lstm_summary, ens_summary]

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY — ADVANCED MODELS v2")
    print("=" * 60)
    df_results = pd.DataFrame(results)[["model","mean_Sharpe","std_Sharpe","mean_RMSE","mean_R2"]]
    print(df_results.to_string(index=False))

    df_results.to_csv(os.path.join("outputs", "advanced_model_results_v2.csv"), index=False)
    print(f"\n  Summary saved → outputs/advanced_model_results_v2.csv")

    plot_model_comparison(results)

    print("\n" + "█" * 70)
    print("  Advanced modelling v2 complete.")
    print("█" * 70 + "\n")

    return df_results

if __name__ == "__main__":
    df_results = main()