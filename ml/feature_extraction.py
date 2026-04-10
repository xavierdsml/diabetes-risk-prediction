"""Feature importance extraction and visualization."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RANDOM_STATE


def extract_feature_importance(X_train, y_train, feature_names):
    """Train a Random Forest and extract feature importances.

    Args:
        X_train: Training features (numpy array).
        y_train: Training labels.
        feature_names: List of feature name strings.

    Returns:
        Dict of {feature_name: importance} sorted by importance descending.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    importances = rf.feature_importances_
    importance_dict = dict(zip(feature_names, importances))

    # Sort descending by importance
    importance_dict = dict(
        sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    )
    return importance_dict


def plot_feature_importance(importances, save_path):
    """Create and save a horizontal bar chart of feature importances.

    Args:
        importances: Dict of {feature_name: importance_value}.
        save_path: File path to save the plot.
    """
    # Sort ascending for horizontal bar chart (top feature at top)
    sorted_items = sorted(importances.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(features)))
    ax.barh(features, values, color=colors)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for i, v in enumerate(values):
        ax.text(v + 0.002, i, f'{v:.4f}', va='center', fontsize=10)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Feature importance plot saved to {save_path}")


if __name__ == '__main__':
    from ml.dataset import load_dataset
    from ml.preprocessing import preprocess_pipeline
    from config import FEATURE_COLUMNS, PLOTS_DIR

    df = load_dataset()
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)
    importances = extract_feature_importance(X_train, y_train, FEATURE_COLUMNS)
    print("Feature importances:")
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")
    plot_feature_importance(importances, os.path.join(PLOTS_DIR, 'feature_importance.png'))
