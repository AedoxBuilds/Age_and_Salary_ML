"""
STAGE 1 - Load data, preprocess, save CSV.

Reusable: works with ANY csv.
    python preprocess.py                       -> generates the 20-person sample dataset and preprocesses it
    python preprocess.py --input my.csv        -> preprocesses any CSV (numeric columns auto-detected)
    python preprocess.py --input my.csv --columns age salary --output data/clean.csv
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

RAW_PATH = "data/people_raw.csv"
CLEAN_PATH = "data/people_preprocessed.csv"
SCALER_PATH = "models/scaler.joblib"


def create_sample_dataset(path=RAW_PATH, seed=42):
    """Task 1: age + salary of 20 people (3 natural groups + 2 outliers)."""
    rng = np.random.default_rng(seed)
    young = np.column_stack([rng.integers(22, 29, 6), rng.integers(40, 70, 6) * 1000])
    mid = np.column_stack([rng.integers(32, 42, 6), rng.integers(120, 170, 6) * 1000])
    senior = np.column_stack([rng.integers(48, 60, 6), rng.integers(260, 330, 6) * 1000])
    outliers = np.array([[25, 400000], [58, 30000]])  # unusual people -> DBSCAN noise
    data = np.vstack([young, mid, senior, outliers])
    df = pd.DataFrame(data, columns=["age", "salary"])
    df.insert(0, "person_id", range(1, len(df) + 1))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_csv(path):
    return pd.read_csv(path)


def preprocess(df, columns=None, id_like_threshold=0.95):
    """Generic cleaning: drop duplicates, pick numeric cols, impute, scale.
    Returns (clean_df, scaler, used_columns). Original columns are kept; scaled
    versions are added with a 'scaled_' prefix."""
    df = df.drop_duplicates().reset_index(drop=True)

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()
        # drop ID-like columns (almost every value unique & integer-sequential)
        columns = [c for c in columns
                   if not (c.lower().endswith("id") or c.lower() == "id")]
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in CSV: {missing}")

    # impute missing numeric values with the median
    for c in columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df[c] = df[c].fillna(df[c].median())

    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[columns])
    for i, c in enumerate(columns):
        df[f"scaled_{c}"] = scaled[:, i]
    return df, scaler, columns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="path to any CSV (default: generate sample)")
    ap.add_argument("--columns", nargs="*", default=None, help="numeric columns to use")
    ap.add_argument("--output", default=CLEAN_PATH)
    args = ap.parse_args()

    df = load_csv(args.input) if args.input else create_sample_dataset()
    print("Raw data:\n", df.head(), "\n")

    clean, scaler, cols = preprocess(df, args.columns)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    os.makedirs("models", exist_ok=True)
    clean.to_csv(args.output, index=False)
    joblib.dump({"scaler": scaler, "columns": cols}, SCALER_PATH)
    print(f"Features used: {cols}")
    print(f"Saved preprocessed data -> {args.output}")
    print(f"Saved scaler            -> {SCALER_PATH}")


if __name__ == "__main__":
    main()