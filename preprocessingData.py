"""
CS 6140 - Hull Tactical Market Prediction
Data Preprocessing Script
Author: Ankit Amonkar

Responsibilities:
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


# CONFIGURATION  — change only these paths to point at different CSVs
TRAIN_PATH = "train.csv"
TEST_PATH  = "test.csv"

TARGET_COL            = "market_forward_excess_returns"  # competition target
DROP_COLS_TRAIN_ONLY  = ["forward_returns", "risk_free_rate"]  # leakage cols
ID_COLS               = ["row_id", "date_id"]                  # non-feature IDs

OUTPUT_DIR  = "outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


# SECTION 1 — DATA LOADING
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test CSVs with basic validation."""
    print("=" * 70)
    print("SECTION 1 — DATA LOADING")
    print("=" * 70)

    train = pd.read_csv(train_path)
    test  = pd.read_csv(test_path)

    print(f"  Train shape : {train.shape}")
    print(f"  Test  shape : {test.shape}")

    # Columns present in one but not the other (besides row_id / target)
    only_train = set(train.columns) - set(test.columns)
    only_test  = set(test.columns)  - set(train.columns)
    if only_train:
        print(f"  Columns only in train : {only_train}")
    if only_test:
        print(f"  Columns only in test  : {only_test}")

    return train, test


# MAIN PIPELINE

def main():
    print("\n" + "█" * 70)
    print("  CS 6140 — Hull Tactical Market Prediction  |  Preprocessing Pipeline")
    print("█" * 70 + "\n")

    # 1. Load
    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)


    return train_raw, test_raw


if __name__ == "__main__":
    train_out, test_out, metadata, scaler = main()
