"""Model evaluation, metrics computation, and visualization."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def evaluate_model(model, X_test, y_test):
    """Evaluate a single model and return a dict of metrics.

    Args:
        model: Trained sklearn-compatible classifier.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc_roc,
        confusion_matrix, fpr, tpr.
    """
    y_pred = model.predict(X_test)

    # Get probabilities for ROC curve
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_proba = model.decision_function(X_test)

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    results = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred, zero_division=0)),
        'f1': float(f1_score(y_test, y_pred, zero_division=0)),
        'auc_roc': float(roc_auc_score(y_test, y_proba)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
    }
    return results


def evaluate_all(models_dict, X_test, y_test):
    """Evaluate all models and return a dict of results keyed by model name.

    Args:
        models_dict: Dict of {model_name: trained_model}.
        X_test: Test features.
        y_test: Test labels.

    Returns:
        Dict of {model_name: metrics_dict}.
    """
    results = {}
    for name, model in models_dict.items():
        results[name] = evaluate_model(model, X_test, y_test)
        print(f"  {name}: accuracy={results[name]['accuracy']:.4f}, "
              f"AUC={results[name]['auc_roc']:.4f}")
    return results


def plot_confusion_matrices(results, save_path):
    """Plot a grid of confusion matrices for all models.

    Args:
        results: Dict of {model_name: metrics_dict} from evaluate_all.
        save_path: File path to save the plot.
    """
    model_names = list(results.keys())
    n = len(model_names)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for i, name in enumerate(model_names):
        cm = np.array(results[name]['confusion_matrix'])
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'],
        )
        axes[i].set_title(name, fontsize=11, fontweight='bold')
        axes[i].set_ylabel('Actual')
        axes[i].set_xlabel('Predicted')

    # Hide unused subplots
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('Confusion Matrices', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrices plot saved to {save_path}")


def plot_roc_curves(results, save_path):
    """Plot all ROC curves on a single figure.

    Args:
        results: Dict of {model_name: metrics_dict}.
        save_path: File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = plt.cm.tab10(np.linspace(0, 1, len(results)))
    for (name, metrics), color in zip(results.items(), colors):
        fpr = metrics['fpr']
        tpr = metrics['tpr']
        auc = metrics['auc_roc']
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"ROC curves plot saved to {save_path}")


def plot_metrics_comparison(results, save_path):
    """Grouped bar chart comparing accuracy, precision, recall, and F1 across models.

    Args:
        results: Dict of {model_name: metrics_dict}.
        save_path: File path to save the plot.
    """
    metrics_keys = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(results.keys())

    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(model_names))
    width = 0.18
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

    for i, metric in enumerate(metrics_keys):
        values = [results[name][metric] for name in model_names]
        bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, rotation=45
            )

    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Metrics Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=30, ha='right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Metrics comparison plot saved to {save_path}")


def plot_correlation_heatmap(df, save_path):
    """Plot a correlation heatmap of all numeric features.

    Args:
        df: DataFrame with numeric features.
        save_path: File path to save the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
        center=0, ax=ax, square=True, linewidths=0.5,
        vmin=-1, vmax=1,
    )
    ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Correlation heatmap saved to {save_path}")


def save_metrics_json(results, save_path):
    """Save metrics to a JSON file, excluding non-serializable items (fpr, tpr, cm).

    Args:
        results: Dict of {model_name: metrics_dict}.
        save_path: File path for the JSON output.
    """
    serializable = {}
    exclude_keys = {'fpr', 'tpr', 'confusion_matrix'}

    for model_name, metrics in results.items():
        serializable[model_name] = {
            k: round(v, 5) if isinstance(v, float) else v
            for k, v in metrics.items()
            if k not in exclude_keys
        }

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics JSON saved to {save_path}")
