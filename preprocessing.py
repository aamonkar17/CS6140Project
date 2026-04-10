"""
CS 6140 - Hull Tactical Market Prediction
Data Preprocessing Script
  - Data acquisition & loading
  - Exploratory Data Analysis (EDA)
  - Missing value handling
  - Feature scaling & normalization
  - Visualizations (correlation matrix, distributions)

The script is designed to be CSV-agnostic: point TRAIN_PATH / TEST_PATH
at any similarly structured files and the pipeline will adapt automatically.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# Change these paths to point at different CSVs
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

TARGET_COL            = "market_forward_excess_returns"
DROP_COLS_TRAIN_ONLY  = ["forward_returns", "risk_free_rate"]  # leakage cols — never feed into the model
ID_COLS               = ["row_id", "date_id"]

OUTPUT_DIR  = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# SECTION 1 — DATA LOADING
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read both CSVs and do a quick sanity check on column alignment.
    Any columns that exist in only one file get printed out — usually
    that's just the target and leakage cols, but worth confirming.
    """
    print("=" * 70)
    print("SECTION 1 — DATA LOADING")
    print("=" * 70)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    print(f"  Train shape : {train.shape}")
    print(f"  Test  shape : {test.shape}")

    only_train = set(train.columns) - set(test.columns)
    only_test  = set(test.columns)  - set(train.columns)
    if only_train:
        print(f"  Columns only in train : {only_train}")
    if only_test:
        print(f"  Columns only in test  : {only_test}")

    return train, test

# SECTION 2 — EXPLORATORY DATA ANALYSIS (EDA)
def run_eda(train: pd.DataFrame, target: str) -> dict:
    """
    Walk through the training data and build up a metadata dict that the rest
    of the pipeline depends on. Covers descriptive stats, column type detection
    (IDs, binary flags, continuous features, target, leakage), and missing value
    rates bucketed into three tiers. Returns the dict so downstream functions
    don't have to re-infer any of this.
    """
    print("\n" + "=" * 70)
    print("SECTION 2 — EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    print("\n[2a] Descriptive statistics (numeric columns)\n")
    print(train.describe().T.to_string())

    # Figure out which columns play which role
    all_cols       = train.columns.tolist()
    id_present     = [c for c in ID_COLS if c in all_cols]
    target_present = [target] if target in all_cols else []
    drop_present   = [c for c in DROP_COLS_TRAIN_ONLY if c in all_cols]

    # Binary cols are integer-typed and only ever take values in {-1, 0, 1}
    binary_cols = [
        c for c in train.select_dtypes(include=[np.integer]).columns
        if c not in id_present
        and set(train[c].dropna().unique()).issubset({-1, 0, 1})
    ]

    numeric_cols = [
        c for c in train.select_dtypes(include=[np.number]).columns
        if c not in id_present + binary_cols + target_present + drop_present
    ]

    print(f"\n[2b] Column taxonomy")
    print(f"  ID / metadata cols      : {id_present}")
    print(f"  Binary / flag cols      : {binary_cols}")
    print(f"  Continuous feature cols : {len(numeric_cols)}")
    print(f"  Target col              : {target_present}")
    print(f"  Leakage cols (drop)     : {drop_present}")

    miss_pct = train.isnull().mean().mul(100).sort_values(ascending=False)
    miss_pct = miss_pct[miss_pct > 0]

    print(f"\n[2c] Missing values — {len(miss_pct)} columns have nulls")
    print(miss_pct.to_string())

    high_miss = miss_pct[miss_pct >= 50].index.tolist()
    med_miss  = miss_pct[(miss_pct >= 20) & (miss_pct < 50)].index.tolist()
    low_miss  = miss_pct[(miss_pct > 0)  & (miss_pct < 20)].index.tolist()

    print(f"\n  ≥50% missing   → {len(high_miss)} cols → drop column")
    print(f"  20-50% missing → {len(med_miss)} cols → median + null flag")
    print(f"  <20% missing   → {len(low_miss)} cols → median impute")

    if target in train.columns:
        tgt = train[target].dropna()
        print(f"\n[2d] Target ({target})")
        print(f"  mean={tgt.mean():.6f}  std={tgt.std():.6f}  "
              f"skew={tgt.skew():.3f}  kurtosis={tgt.kurtosis():.3f}")
        print(f"  min={tgt.min():.6f}  max={tgt.max():.6f}")

    meta = {
        "id_cols"     : id_present,
        "binary_cols" : binary_cols,
        "numeric_cols": numeric_cols,
        "target_col"  : target_present[0] if target_present else None,
        "drop_cols"   : drop_present,
        "miss_high"   : high_miss,
        "miss_med"    : med_miss,
        "miss_low"    : low_miss,
        "miss_pct"    : miss_pct,
    }
    return meta

# SECTION 3 — MISSING VALUE HANDLING
def handle_missing(train: pd.DataFrame, test: pd.DataFrame,
                   meta: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Three-tier imputation based on how much data is missing per column.
    Columns with ≥50% nulls just get dropped — not worth keeping. The middle
    tier (20-50%) gets median-imputed plus a binary flag so the model can learn
    from the missingness pattern itself. Anything under 20% is a simple median
    fill. All medians come from the training set so nothing leaks into test.
    """
    print("\n" + "=" * 70)
    print("SECTION 3 — MISSING VALUE HANDLING")
    print("=" * 70)

    train = train.copy()
    test  = test.copy()

    # Drop anything with ≥50% missing (skip protected cols)
    protected = meta["id_cols"] + [meta["target_col"]] + meta["drop_cols"]
    cols_to_drop = [c for c in meta["miss_high"] if c in train.columns and c not in protected]
    train.drop(columns=cols_to_drop, inplace=True)
    test.drop(columns=[c for c in cols_to_drop if c in test.columns], inplace=True)
    print(f"\n  Dropped {len(cols_to_drop)} columns with ≥50% missing: {cols_to_drop}")

    # Median impute + add a was_null flag (20–50% missing)
    med_cols = [c for c in meta["miss_med"] if c in train.columns and c not in protected]
    medians_med = train[med_cols].median()
    for col in med_cols:
        flag_col = f"{col}_was_null"
        train[flag_col] = train[col].isnull().astype(int)
        if col in test.columns:
            test[flag_col] = test[col].isnull().astype(int)
        train[col].fillna(medians_med[col], inplace=True)
        if col in test.columns:
            test[col].fillna(medians_med[col], inplace=True)
    print(f"\n  Median-imputed + null flag for {len(med_cols)} cols (20–50% missing)")

    # Simple median impute (<20% missing)
    low_cols = [c for c in meta["miss_low"] if c in train.columns and c not in protected]
    medians_low = train[low_cols].median()
    train[low_cols] = train[low_cols].fillna(medians_low)
    for col in low_cols:
        if col in test.columns:
            test[col].fillna(medians_low[col], inplace=True)
    print(f"  Median-imputed {len(low_cols)} cols (<20% missing)")

    remaining = train.isnull().sum().sum()
    print(f"\n  Remaining nulls in train after imputation: {remaining}")

    return train, test

# SECTION 4 — FEATURE SCALING & NORMALIZATION
def scale_features(train: pd.DataFrame, test: pd.DataFrame,
                   meta: dict, method: str = "robust") -> tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Scale continuous numeric features only — binary cols, null flags, IDs,
    the target, and known leakage cols are all left alone. Scaler is fit on
    train and applied to test to avoid leakage. The fitted scaler is returned
    so it can be reused at inference time without re-running this function.

    method options:
      'robust'   — RobustScaler  (median + IQR; holds up well against financial outliers)
      'standard' — StandardScaler (z-score)
      'minmax'   — MinMaxScaler  (squashes everything to [0, 1])
    """
    print("\n" + "=" * 70)
    print("SECTION 4 — FEATURE SCALING & NORMALIZATION")
    print("=" * 70)
    print(f"\n  Scaling method : {method}")

    scalers = {"robust": RobustScaler, "standard": StandardScaler, "minmax": MinMaxScaler}
    if method not in scalers:
        raise ValueError(f"method must be one of {list(scalers.keys())}")

    scaler = scalers[method]()

    # Build exclusion list — everything that shouldn't be scaled
    was_null_flags = [c for c in train.columns if c.endswith("_was_null")]
    exclude = (
        meta["id_cols"]
        + meta["binary_cols"]
        + ([meta["target_col"]] if meta["target_col"] else [])
        + meta["drop_cols"]
        + meta["miss_high"]  # already dropped, but harmless to list
        + was_null_flags
    )

    scale_cols = [
        c for c in train.columns
        if c not in exclude and c in train.select_dtypes(include=[np.number]).columns
    ]

    train = train.copy()
    test  = test.copy()

    train[scale_cols] = scaler.fit_transform(train[scale_cols])
    test_scale_cols   = [c for c in scale_cols if c in test.columns]
    test[test_scale_cols] = scaler.transform(test[test_scale_cols])

    print(f"  Scaled {len(scale_cols)} continuous feature columns")
    print(f"  Left unchanged: {meta['binary_cols'] + was_null_flags}")

    return train, test, scaler

# SECTION 5 — VISUALIZATIONS
def plot_missing_heatmap(miss_pct: pd.Series, save_dir: str) -> None:
    """
    Horizontal bar chart of missing rates, color-coded by tier (red ≥50%,
    orange 20-50%, blue <20%). Threshold lines at 20% and 50% make it easy
    to see at a glance which columns are going to be dropped vs flagged.
    Does nothing if there are no missing values.
    """
    if miss_pct.empty:
        return

    fig, ax = plt.subplots(figsize=(14, max(5, len(miss_pct) * 0.3)))
    colors = ["#e74c3c" if v >= 50 else "#e67e22" if v >= 20 else "#3498db"
              for v in miss_pct.values]
    miss_pct.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.2, label="50% threshold (drop)")
    ax.axvline(20, color="#e67e22", linestyle="--", linewidth=1.2, label="20% threshold (flag)")
    ax.set_xlabel("% Missing")
    ax.set_title("Missing Value Rates by Column", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(save_dir, "missing_values.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

def plot_target_distribution(train: pd.DataFrame, target: str, save_dir: str) -> None:
    """
    Side-by-side histogram (with mean/median lines) and Q-Q plot for the target.
    The Q-Q plot is a quick check on normality — financial return series often
    have fat tails, and this will show that clearly. Skips if target col is absent.
    """
    if target not in train.columns:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    tgt = train[target].dropna()

    axes[0].hist(tgt, bins=60, color="#3498db", edgecolor="white", alpha=0.85)
    axes[0].set_title(f"Distribution of {target}", fontweight="bold")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("Frequency")
    axes[0].axvline(tgt.mean(), color="red", linestyle="--", label=f"Mean: {tgt.mean():.4f}")
    axes[0].axvline(tgt.median(), color="green", linestyle="--", label=f"Median: {tgt.median():.4f}")
    axes[0].legend()

    from scipy import stats
    stats.probplot(tgt, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot (Target vs Normal)", fontweight="bold")

    plt.suptitle("Target Variable Analysis", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_feature_distributions(train: pd.DataFrame, feature_cols: list,
                                save_dir: str, max_cols: int = 20) -> None:
    """
    Grid of histograms for the first max_cols continuous features. Good for
    spotting skew or scale issues before and after normalization. Any leftover
    subplot cells in the grid are hidden so the layout stays clean.
    """
    cols = [c for c in feature_cols if c in train.columns][:max_cols]
    if not cols:
        return

    n     = len(cols)
    ncols = 5
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(train[col].dropna(), bins=40, color="#2ecc71", edgecolor="white", alpha=0.8)
        axes[i].set_title(col, fontsize=9)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle(f"Feature Distributions (first {n} continuous features)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(save_dir, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_correlation_matrix(train: pd.DataFrame, feature_cols: list,
                             target: str, save_dir: str,
                             top_n: int = 30) -> None:
    """
    Lower-triangle correlation heatmap for the top_n features by absolute
    correlation with the target. Capping at top_n keeps things readable — a
    full-dataset heatmap with 100+ features is basically unreadable anyway.
    """
    cols = [c for c in feature_cols if c in train.columns]
    if target in train.columns:
        corr_with_target = (
            train[cols + [target]]
            .corr()[target]
            .drop(target)
            .abs()
            .sort_values(ascending=False)
        )
        selected = corr_with_target.head(top_n).index.tolist() + [target]
    else:
        selected = cols[:top_n]

    corr_mat = train[selected].corr()

    fig, ax = plt.subplots(figsize=(16, 14))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(
        corr_mat,
        mask=mask,
        annot=False,
        cmap="RdBu_r",
        center=0,
        vmin=-1, vmax=1,
        linewidths=0.3,
        ax=ax,
        cbar_kws={"shrink": 0.8}
    )
    ax.set_title(
        f"Correlation Matrix — Top {top_n} features by |corr| with target",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(save_dir, "correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

def plot_top_feature_correlations(train: pd.DataFrame, feature_cols: list,
                                   target: str, save_dir: str,
                                   top_n: int = 20) -> None:
    """
    Horizontal bar chart ranking features by absolute Pearson correlation with
    the target. Green = positive, red = negative, so directionality is obvious
    at a glance. Skips quietly if the target column isn't in the DataFrame.
    """
    if target not in train.columns:
        return

    cols = [c for c in feature_cols if c in train.columns]
    corrs = (
        train[cols + [target]]
        .corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
        .head(top_n)
    )

    fig, ax = plt.subplots(figsize=(10, max(5, top_n * 0.4)))
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in corrs.values]
    corrs.sort_values().plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Top {top_n} Features by Correlation with Target",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Pearson Correlation")
    plt.tight_layout()
    path = os.path.join(save_dir, "top_feature_correlations.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

def plot_binary_feature_counts(train: pd.DataFrame, binary_cols: list,
                                save_dir: str) -> None:
    """
    Grouped bar chart of value counts for each binary/flag column. Useful for
    checking that {-1, 0, 1} flags aren't wildly imbalanced before modelling.
    Does nothing if there are no binary columns.
    """
    if not binary_cols:
        return

    counts = {col: train[col].value_counts().to_dict() for col in binary_cols}
    df_counts = pd.DataFrame(counts).T.fillna(0)

    fig, ax = plt.subplots(figsize=(12, max(4, len(binary_cols) * 0.6)))
    df_counts.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white")
    ax.set_title("Binary / Flag Feature Value Counts", fontsize=13, fontweight="bold")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Value")
    plt.tight_layout()
    path = os.path.join(save_dir, "binary_feature_counts.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved → {path}")

def run_visualizations(train_raw: pd.DataFrame, train_scaled: pd.DataFrame,
                        meta: dict) -> None:
    """
    Calls all the individual plot functions in order. Raw train is used for the
    target distribution and binary counts (no point showing scaled values there),
    while the scaled version is used for feature distributions and correlations
    since that's what the model will actually see.
    """
    print("\n" + "=" * 70)
    print("SECTION 5 — VISUALIZATIONS")
    print("=" * 70)

    sd        = FIGURES_DIR
    feat_cols = meta["numeric_cols"]

    plot_missing_heatmap(meta["miss_pct"], sd)
    plot_target_distribution(train_raw, meta["target_col"] or TARGET_COL, sd)
    plot_feature_distributions(train_scaled, feat_cols, sd, max_cols=20)
    plot_correlation_matrix(train_scaled, feat_cols, meta["target_col"] or TARGET_COL, sd, top_n=30)
    plot_top_feature_correlations(train_scaled, feat_cols, meta["target_col"] or TARGET_COL, sd, top_n=20)
    plot_binary_feature_counts(train_raw, meta["binary_cols"], sd)

# SECTION 6 — SAVE PREPROCESSED DATA
def save_preprocessed(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """
    Dump the final preprocessed train and test sets to CSV. These files are
    what the modelling scripts should be pointed at — everything upstream of
    this has already been handled.
    """
    print("\n" + "=" * 70)
    print("SECTION 6 — SAVING PREPROCESSED DATA")
    print("=" * 70)

    train_path = os.path.join(OUTPUT_DIR, "train_preprocessed.csv")
    test_path  = os.path.join(OUTPUT_DIR, "test_preprocessed.csv")

    train.to_csv(train_path, index=False)
    test.to_csv(test_path,  index=False)

    print(f"\n  Train preprocessed shape : {train.shape}  →  {train_path}")
    print(f"  Test  preprocessed shape : {test.shape}   →  {test_path}")

# MAIN PIPELINE
def main():
    """
    Runs the full preprocessing pipeline end to end: load, EDA, imputation,
    scaling, visualizations, save. Returns the scaled DataFrames, metadata dict,
    and fitted scaler so the caller can use them without re-running anything.
    """
    print("\n" + "█" * 70)
    print("  CS 6140 — Hull Tactical Market Prediction  |  Preprocessing Pipeline")
    print("█" * 70 + "\n")

    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)
    meta = run_eda(train_raw, TARGET_COL)
    train_clean, test_clean = handle_missing(train_raw, test_raw, meta)

    # Switch method to 'standard' or 'minmax' if needed
    train_scaled, test_scaled, fitted_scaler = scale_features(
        train_clean, test_clean, meta, method="robust"
    )

    # Raw train for target/binary plots, scaled for everything else
    run_visualizations(train_raw, train_scaled, meta)

    save_preprocessed(train_scaled, test_scaled)

    print("\n" + "█" * 70)
    print("  Preprocessing complete.  Outputs in:", OUTPUT_DIR)
    print("█" * 70 + "\n")

    return train_scaled, test_scaled, meta, fitted_scaler

if __name__ == "__main__":
    train_out, test_out, metadata, scaler = main()