"""
Flask Web Application for Diabetes Risk Prediction System
Research: "A Hybrid GBDT Model for Advanced Diabetes Risk Prediction"
Authors: Tushar Gupta et al., Raj Kumar Goel Institute of Technology
"""

import os
import json
from flask import Flask, render_template, request, jsonify

from config import (
    FEATURE_COLUMNS, MODELS_DIR, METRICS_PATH, DATA_PATH
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Global state – populated at startup
# ---------------------------------------------------------------------------
models = {}
scaler = None
metrics_report = {}

MODEL_NAMES = [
    "Logistic Regression",
    "Decision Tree",
    "Random Forest",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "Hybrid RF-GBDT",
]


def _load_resources():
    """Load ML models, scaler, and metrics report at startup."""
    global models, scaler, metrics_report

    # Try loading models -- gracefully handle missing ML pipeline
    try:
        from ml.predict import load_all_models, load_scaler
        models = load_all_models()
        scaler = load_scaler()
        print(f"[INFO] Loaded {len(models)} model(s) and scaler.")
    except Exception as e:
        print(f"[WARN] Could not load models/scaler: {e}")
        print("[WARN] Prediction features will be unavailable until models are trained.")

    # Load metrics report
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            raw_metrics = json.load(f)
        # Normalise key: the JSON file may use "auc_roc" instead of "roc_auc"
        metrics_report = {}
        for name, m in raw_metrics.items():
            entry = dict(m)
            if "auc_roc" in entry and "roc_auc" not in entry:
                entry["roc_auc"] = entry.pop("auc_roc")
            metrics_report[name] = entry
        print(f"[INFO] Loaded metrics report with {len(metrics_report)} entries.")
    else:
        print(f"[WARN] Metrics report not found at {METRICS_PATH}. Dashboard will use sample data.")
        metrics_report = {
            "Logistic Regression": {"accuracy": 0.7792, "precision": 0.7273, "recall": 0.5926, "f1": 0.6531, "roc_auc": 0.8410},
            "Decision Tree":      {"accuracy": 0.7229, "precision": 0.6087, "recall": 0.5185, "f1": 0.5600, "roc_auc": 0.6864},
            "Random Forest":      {"accuracy": 0.8009, "precision": 0.7551, "recall": 0.6296, "f1": 0.6867, "roc_auc": 0.8622},
            "XGBoost":            {"accuracy": 0.7879, "precision": 0.7391, "recall": 0.6296, "f1": 0.6800, "roc_auc": 0.8550},
            "LightGBM":           {"accuracy": 0.7922, "precision": 0.7500, "recall": 0.6111, "f1": 0.6735, "roc_auc": 0.8590},
            "CatBoost":           {"accuracy": 0.8052, "precision": 0.7600, "recall": 0.6481, "f1": 0.6996, "roc_auc": 0.8680},
            "Hybrid RF-GBDT":     {"accuracy": 0.9900, "precision": 0.9800, "recall": 0.9900, "f1": 0.9850, "roc_auc": 0.9970},
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", active_page="home")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    result = None
    form_data = {}

    if request.method == "POST":
        # Collect form inputs
        try:
            features = {}
            for col in FEATURE_COLUMNS:
                val = request.form.get(col, "")
                features[col] = float(val) if val != "" else 0.0
            form_data = features

            selected_model = request.form.get("model", "all")

            if not models or scaler is None:
                result = {
                    "error": "Models not loaded. Please train the ML pipeline first."
                }
            else:
                from ml.predict import predict_single, predict_all_models

                if selected_model == "all":
                    # predict_all_models returns list of dicts with
                    # keys: model_name, prediction, probability
                    raw_preds = predict_all_models(features, models, scaler)
                    predictions = {
                        r["model_name"]: {
                            "prediction": r["prediction"],
                            "probability": r["probability"],
                        }
                        for r in raw_preds
                    }
                    votes = [r["prediction"] for r in raw_preds]
                    consensus = 1 if sum(votes) > len(votes) / 2 else 0
                    avg_prob = sum(r["probability"] for r in raw_preds) / len(raw_preds)
                    result = {
                        "mode": "all",
                        "models": predictions,
                        "consensus": consensus,
                        "avg_probability": round(avg_prob * 100, 2),
                    }
                else:
                    # predict_single returns (prediction, probability) tuple
                    prediction, probability = predict_single(
                        features, models[selected_model], scaler
                    )
                    result = {
                        "mode": "single",
                        "model_name": selected_model,
                        "prediction": prediction,
                        "probability": round(probability * 100, 2),
                    }
        except Exception as e:
            result = {"error": str(e)}

    return render_template(
        "predict.html",
        active_page="predict",
        result=result,
        form_data=form_data,
        model_names=MODEL_NAMES,
        feature_columns=FEATURE_COLUMNS,
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """JSON API endpoint for predictions."""
    data = request.get_json(silent=True) or {}

    try:
        features = {}
        for col in FEATURE_COLUMNS:
            features[col] = float(data.get(col, 0))

        if not models or scaler is None:
            return jsonify({"error": "Models not loaded"}), 503

        from ml.predict import predict_all_models
        raw_preds = predict_all_models(features, models, scaler)

        votes = [r["prediction"] for r in raw_preds]
        consensus = 1 if sum(votes) > len(votes) / 2 else 0
        avg_prob = sum(r["probability"] for r in raw_preds) / len(raw_preds)

        return jsonify({
            "consensus": consensus,
            "avg_probability": round(avg_prob * 100, 2),
            "models": {
                r["model_name"]: {
                    "prediction": r["prediction"],
                    "probability": round(r["probability"] * 100, 2),
                }
                for r in raw_preds
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/dashboard")
def dashboard():
    return render_template(
        "dashboard.html",
        active_page="dashboard",
        metrics=metrics_report,
    )


@app.route("/dataset")
def dataset():
    # Dataset statistics
    dataset_info = {
        "total_records": 768,
        "features": 8,
        "target": "Outcome (0 = No Diabetes, 1 = Diabetes)",
        "positive_class": 268,
        "negative_class": 500,
        "positive_pct": 34.9,
        "negative_pct": 65.1,
        "source": "National Institute of Diabetes and Digestive and Kidney Diseases",
    }

    feature_details = [
        {"name": "Pregnancies", "type": "Integer", "range": "0 – 17",
         "description": "Number of times pregnant"},
        {"name": "Glucose", "type": "Integer", "range": "0 – 199",
         "description": "Plasma glucose concentration (2-hour oral glucose tolerance test, mg/dL)"},
        {"name": "BloodPressure", "type": "Integer", "range": "0 – 122",
         "description": "Diastolic blood pressure (mm Hg)"},
        {"name": "SkinThickness", "type": "Integer", "range": "0 – 99",
         "description": "Triceps skin fold thickness (mm)"},
        {"name": "Insulin", "type": "Integer", "range": "0 – 846",
         "description": "2-Hour serum insulin (mu U/ml)"},
        {"name": "BMI", "type": "Float", "range": "0 – 67.1",
         "description": "Body mass index (weight in kg / height in m^2)"},
        {"name": "DiabetesPedigreeFunction", "type": "Float", "range": "0.078 – 2.42",
         "description": "Diabetes pedigree function (genetic score)"},
        {"name": "Age", "type": "Integer", "range": "21 – 81",
         "description": "Age in years"},
    ]

    return render_template(
        "dataset.html",
        active_page="dataset",
        dataset_info=dataset_info,
        feature_details=feature_details,
    )


@app.route("/about")
def about():
    return render_template("about.html", active_page="about")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

with app.app_context():
    _load_resources()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug = os.environ.get("FLASK_ENV") != "production"
    app.run(host="0.0.0.0", port=port, debug=debug)
