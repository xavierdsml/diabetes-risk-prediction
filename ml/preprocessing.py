"""Preprocessing utilities: imputation, normalization, and full pipeline."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (
    FEATURE_COLUMNS, TARGET_COLUMN, ZERO_IMPUTE_COLUMNS,
    TEST_SIZE, RANDOM_STATE,
)


def impute_zeros(df, columns=None):
    """Replace biologically impossible zeros with NaN, then fill with column mean.

    Args:
        df: DataFrame to process (modified in-place on a copy).
        columns: List of columns to impute. Defaults to ZERO_IMPUTE_COLUMNS.

    Returns:
        DataFrame with zeros replaced by column means in specified columns.
    """
    df = df.copy()
    cols = columns or ZERO_IMPUTE_COLUMNS
    for col in cols:
        if col in df.columns:
            df[col] = df[col].replace(0, np.nan)
            col_mean = df[col].mean()
            df[col] = df[col].fillna(col_mean)
    return df


def normalize(X_train, X_test):
    """Fit MinMaxScaler on training data and transform both train and test.

    Args:
        X_train: Training feature array or DataFrame.
        X_test: Test feature array or DataFrame.

    Returns:
        Tuple of (X_train_scaled, X_test_scaled, scaler).
    """
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_pipeline(df):
    """Full preprocessing pipeline: impute zeros, split, and normalize.

    Args:
        df: Raw DataFrame with features and target.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test, scaler).
        X_train and X_test are numpy arrays (scaled).
    """
    # Step 1: Impute biologically impossible zeros
    df = impute_zeros(df)

    # Step 2: Separate features and target
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # Step 3: Train-test split (70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Step 4: Normalize features
    X_train_scaled, X_test_scaled, scaler = normalize(X_train, X_test)

    print(f"Preprocessing complete: train={X_train_scaled.shape[0]}, test={X_test_scaled.shape[0]}")
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler


if __name__ == '__main__':
    from ml.dataset import load_dataset
    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
