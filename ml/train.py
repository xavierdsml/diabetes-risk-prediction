"""Main training script for the diabetes prediction pipeline.

Run with: python -m ml.train
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from config import (
    FEATURE_COLUMNS, MODELS_DIR, PLOTS_DIR, METRICS_PATH,
)
from ml.dataset import load_dataset, validate_dataset, get_dataset_info
from ml.preprocessing import preprocess_pipeline
from ml.feature_extraction import extract_feature_importance, plot_feature_importance
from ml.models import get_models, get_hybrid_rf_gbdt, train_model, train_hybrid
from ml.evaluate import (
    evaluate_all, plot_confusion_matrices, plot_roc_curves,
    plot_metrics_comparison, plot_correlation_heatmap, save_metrics_json,
)


def main():
    """Execute the full ML training pipeline."""
    print("=" * 60)
    print("  DIABETES PREDICTION - ML PIPELINE")
    print("=" * 60)

    # ---- Step 1: Load and validate dataset ----
    print("\n[1/11] Loading dataset...")
    df = load_dataset()
    validate_dataset(df)
    info = get_dataset_info(df)
    print(f"  Class distribution: {info['class_distribution']}")
    print(f"  Zeros to impute: {info['missing_zeros_count']}")

    # ---- Step 2: Correlation heatmap ----
    print("\n[2/11] Plotting correlation heatmap...")
    plot_correlation_heatmap(df, os.path.join(PLOTS_DIR, 'correlation_heatmap.png'))

    # ---- Step 3: Preprocess (impute + split + normalize) ----
    print("\n[3/11] Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(df)

    # ---- Step 4: Feature importance ----
    print("\n[4/11] Extracting feature importance...")
    importances = extract_feature_importance(X_train, y_train, FEATURE_COLUMNS)
    for feat, imp in importances.items():
        print(f"  {feat}: {imp:.4f}")
    plot_feature_importance(importances, os.path.join(PLOTS_DIR, 'feature_importance.png'))

    # ---- Step 5: Train all 6 models + hybrid ----
    print("\n[5/11] Training models...")
    models = get_models()
    trained_models = {}
    for name, model in models.items():
        print(f"  Training {name}...")
        trained_models[name] = train_model(model, X_train, y_train)

    print("  Training Hybrid RF-GBDT...")
    rf, gbdt = get_hybrid_rf_gbdt()
    hybrid_model = train_hybrid(rf, gbdt, X_train, y_train)
    trained_models['Hybrid RF-GBDT'] = hybrid_model

    # ---- Step 6: Evaluate all models ----
    print("\n[6/11] Evaluating models...")
    results = evaluate_all(trained_models, X_test, y_test)

    # ---- Step 7: Save models ----
    print("\n[7/11] Saving models...")
    os.makedirs(MODELS_DIR, exist_ok=True)
    for name, model in trained_models.items():
        safe_name = name.lower().replace(' ', '_').replace('-', '_')
        model_path = os.path.join(MODELS_DIR, f'{safe_name}.joblib')
        joblib.dump(model, model_path)
        print(f"  Saved {name} -> {model_path}")

    # ---- Step 8: Save scaler ----
    print("\n[8/11] Saving scaler...")
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler -> {scaler_path}")

    # ---- Step 9: Save metrics JSON ----
    print("\n[9/11] Saving metrics report...")
    save_metrics_json(results, METRICS_PATH)

    # ---- Step 10: Generate all plots ----
    print("\n[10/11] Generating plots...")
    plot_confusion_matrices(results, os.path.join(PLOTS_DIR, 'confusion_matrices.png'))
    plot_roc_curves(results, os.path.join(PLOTS_DIR, 'roc_curves.png'))
    plot_metrics_comparison(results, os.path.join(PLOTS_DIR, 'metrics_comparison.png'))

    # ---- Step 11: Print summary table ----
    print("\n[11/11] Training Complete!")
    print("\n" + "=" * 80)
    print(f"{'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC-ROC':>10}")
    print("-" * 80)
    for name, metrics in results.items():
        print(f"{name:<25} {metrics['accuracy']:>10.4f} {metrics['precision']:>10.4f} "
              f"{metrics['recall']:>10.4f} {metrics['f1']:>10.4f} {metrics['auc_roc']:>10.4f}")
    print("=" * 80)

    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['auc_roc'])
    print(f"\nBest model by AUC-ROC: {best_model[0]} ({best_model[1]['auc_roc']:.4f})")
    print(f"\nArtifacts saved:")
    print(f"  Models:  {MODELS_DIR}/")
    print(f"  Plots:   {PLOTS_DIR}/")
    print(f"  Metrics: {METRICS_PATH}")


if __name__ == '__main__':
    main()
