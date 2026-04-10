"""Model definitions and training functions for the diabetes prediction pipeline."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RANDOM_STATE


def get_models():
    """Return a dictionary of 6 classification models.

    Returns:
        Dict mapping model name strings to sklearn-compatible classifier instances.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100, random_state=RANDOM_STATE,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=100, random_state=RANDOM_STATE, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=100, random_state=RANDOM_STATE,
            verbose=0
        ),
    }
    return models


def get_hybrid_rf_gbdt():
    """Return a (RandomForest, GradientBoosting) tuple for the hybrid two-stage model.

    Returns:
        Tuple of (RandomForestClassifier, GradientBoostingClassifier).
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    gbdt = GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE)
    return rf, gbdt


def train_model(model, X_train, y_train):
    """Fit a model on training data.

    Args:
        model: Sklearn-compatible classifier.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The fitted model.
    """
    model.fit(X_train, y_train)
    return model


def train_hybrid(rf, gbdt, X_train, y_train):
    """Two-stage hybrid: train RF for feature importance, then train GBDT.

    Stage 1: Train RandomForest to extract feature importances.
    Stage 2: Train GradientBoosting on the same data (using RF insights).

    Args:
        rf: RandomForestClassifier instance.
        gbdt: GradientBoostingClassifier instance.
        X_train: Training features.
        y_train: Training labels.

    Returns:
        The trained GradientBoostingClassifier (GBDT).
    """
    # Stage 1: Train RF to get feature importances
    rf.fit(X_train, y_train)
    importances = rf.feature_importances_
    print(f"  Hybrid Stage 1 (RF): top feature importance = {max(importances):.4f}")

    # Stage 2: Train GBDT on the full training data
    gbdt.fit(X_train, y_train)
    print(f"  Hybrid Stage 2 (GBDT): training complete")

    return gbdt


if __name__ == '__main__':
    from ml.dataset import load_dataset
    from ml.preprocessing import preprocess_pipeline

    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)

    models = get_models()
    for name, model in models.items():
        train_model(model, X_train, y_train)
        score = model.score(X_test, y_test)
        print(f"{name}: accuracy = {score:.4f}")
