"""Prediction utilities: load models and run inference on new data."""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", message="X.*feature names")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODELS_DIR, FEATURE_COLUMNS


def load_all_models():
    """Load all .joblib model files from saved_models/ directory.

    Returns:
        Dict of {model_name: loaded_model}. Model name is derived from the
        filename (e.g., 'random_forest' from 'random_forest.joblib').

    Raises:
        FileNotFoundError: If MODELS_DIR does not exist.
    """
    if not os.path.exists(MODELS_DIR):
        raise FileNotFoundError(f"Models directory not found: {MODELS_DIR}")

    NAME_MAP = {
        'catboost': 'CatBoost',
        'decision_tree': 'Decision Tree',
        'hybrid_rf_gbdt': 'Hybrid RF-GBDT',
        'lightgbm': 'LightGBM',
        'logistic_regression': 'Logistic Regression',
        'random_forest': 'Random Forest',
        'xgboost': 'XGBoost',
    }
    models = {}
    for fname in sorted(os.listdir(MODELS_DIR)):
        if fname.endswith('.joblib') and fname != 'scaler.joblib':
            key = fname.replace('.joblib', '')
            model_name = NAME_MAP.get(key, key.replace('_', ' ').title())
            models[model_name] = joblib.load(os.path.join(MODELS_DIR, fname))
    return models


def load_scaler():
    """Load the fitted MinMaxScaler from saved_models/scaler.joblib.

    Returns:
        The loaded MinMaxScaler object.

    Raises:
        FileNotFoundError: If scaler.joblib does not exist.
    """
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    return joblib.load(scaler_path)


def predict_single(features_dict, model, scaler):
    """Predict diabetes outcome for a single patient.

    Args:
        features_dict: Dict with keys matching FEATURE_COLUMNS
            (e.g., {'Pregnancies': 6, 'Glucose': 148, ...}).
        model: Trained sklearn-compatible classifier.
        scaler: Fitted MinMaxScaler.

    Returns:
        Tuple of (prediction, probability).
        prediction: 0 or 1.
        probability: float, probability of class 1 (diabetes).
    """
    # Build feature array in the correct column order
    feature_values = [features_dict[col] for col in FEATURE_COLUMNS]
    X = pd.DataFrame([feature_values], columns=FEATURE_COLUMNS)

    # Scale features (preserve DataFrame with column names to avoid sklearn warnings)
    X_scaled = pd.DataFrame(scaler.transform(X), columns=FEATURE_COLUMNS)

    # Predict
    prediction = int(model.predict(X_scaled)[0])

    if hasattr(model, 'predict_proba'):
        probability = float(model.predict_proba(X_scaled)[0][1])
    elif hasattr(model, 'decision_function'):
        # Sigmoid of decision function as proxy
        decision = model.decision_function(X_scaled)[0]
        probability = float(1 / (1 + np.exp(-decision)))
    else:
        probability = float(prediction)

    return prediction, probability


def predict_all_models(features_dict, models, scaler):
    """Run prediction across all loaded models.

    Args:
        features_dict: Dict with keys matching FEATURE_COLUMNS.
        models: Dict of {model_name: trained_model}.
        scaler: Fitted MinMaxScaler.

    Returns:
        List of dicts, each with keys: model_name, prediction, probability.
    """
    results = []
    for model_name, model in models.items():
        prediction, probability = predict_single(features_dict, model, scaler)
        results.append({
            'model_name': model_name,
            'prediction': prediction,
            'probability': round(probability, 4),
        })
    return results


if __name__ == '__main__':
    # Example usage
    models = load_all_models()
    scaler = load_scaler()

    sample = {
        'Pregnancies': 6,
        'Glucose': 148,
        'BloodPressure': 72,
        'SkinThickness': 35,
        'Insulin': 0,
        'BMI': 33.6,
        'DiabetesPedigreeFunction': 0.627,
        'Age': 50,
    }

    print("Sample input:", sample)
    print("\nPredictions:")
    results = predict_all_models(sample, models, scaler)
    for r in results:
        label = "Diabetes" if r['prediction'] == 1 else "No Diabetes"
        print(f"  {r['model_name']:<30} -> {label} (prob={r['probability']:.4f})")
