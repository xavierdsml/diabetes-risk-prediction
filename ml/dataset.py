"""Dataset loading, validation, and info utilities for the diabetes prediction pipeline."""

import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN, ZERO_IMPUTE_COLUMNS


def load_dataset(path=None):
    """Load the diabetes CSV dataset and return a DataFrame.

    Args:
        path: Optional path to CSV file. Defaults to DATA_PATH from config.

    Returns:
        pd.DataFrame with the loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
    """
    filepath = path or DATA_PATH
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    df = pd.read_csv(filepath)
    return df


def validate_dataset(df):
    """Validate the dataset shape and column names.

    Args:
        df: DataFrame to validate.

    Raises:
        ValueError: If the dataset has incorrect shape or missing columns.
    """
    expected_columns = FEATURE_COLUMNS + [TARGET_COLUMN]

    missing_cols = [c for c in expected_columns if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")

    if df.shape[0] == 0:
        raise ValueError("Dataset is empty (0 rows)")

    if df.shape[1] < len(expected_columns):
        raise ValueError(
            f"Expected at least {len(expected_columns)} columns, got {df.shape[1]}"
        )

    # Check for NaN in target column
    if df[TARGET_COLUMN].isna().any():
        raise ValueError("Target column contains NaN values")

    print(f"Dataset validation passed: {df.shape[0]} rows, {df.shape[1]} columns")


def get_dataset_info(df):
    """Return a dictionary with descriptive statistics about the dataset.

    Args:
        df: DataFrame to analyze.

    Returns:
        dict with keys: shape, class_distribution, feature_ranges, missing_zeros_count.
    """
    info = {}
    info['shape'] = {'rows': df.shape[0], 'columns': df.shape[1]}

    # Class distribution
    class_counts = df[TARGET_COLUMN].value_counts().to_dict()
    info['class_distribution'] = {
        'no_diabetes (0)': int(class_counts.get(0, 0)),
        'diabetes (1)': int(class_counts.get(1, 0)),
        'ratio': round(
            class_counts.get(1, 0) / class_counts.get(0, 1), 3
        ),
    }

    # Feature ranges
    feature_ranges = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            feature_ranges[col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': round(float(df[col].mean()), 2),
                'std': round(float(df[col].std()), 2),
            }
    info['feature_ranges'] = feature_ranges

    # Count biologically impossible zeros in relevant columns
    zeros_count = {}
    for col in ZERO_IMPUTE_COLUMNS:
        if col in df.columns:
            zeros_count[col] = int((df[col] == 0).sum())
    info['missing_zeros_count'] = zeros_count

    return info


if __name__ == '__main__':
    df = load_dataset()
    validate_dataset(df)
    info = get_dataset_info(df)
    import json
    print(json.dumps(info, indent=2))
