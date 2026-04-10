# Diabetes Risk Prediction System

A machine learning-powered diabetes risk prediction system built on a novel **Hybrid RF-GBDT** (Random Forest + Gradient Boosting Decision Tree) ensemble model. Compares 7 state-of-the-art classification algorithms with an interactive web dashboard.

[![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-black?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-blue)](https://xgboost.readthedocs.io)
[![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple?logo=bootstrap&logoColor=white)](https://getbootstrap.com)

---

## Overview

Diabetes mellitus affects over **537 million adults worldwide** and early detection is critical for prevention. This project implements a comprehensive ML pipeline that:

- Trains and compares **7 classification models** on the PIMA Indians Diabetes Dataset
- Proposes a novel **Hybrid RF-GBDT** ensemble combining bagging + boosting
- Provides a **web-based prediction interface** for real-time diabetes risk assessment
- Features an **interactive dashboard** with model comparison charts

> **Research Paper:** *A Hybrid GBDT Model for Advanced Diabetes Risk Prediction*
> Raj Kumar Goel Institute of Technology, Department of CSE (Data Science)

---

## Models Compared

| Model | Type | Accuracy | AUC-ROC | F1 Score |
|-------|------|----------|---------|----------|
| Logistic Regression | Linear | 73.59% | 0.839 | 0.579 |
| Decision Tree | Tree | 70.13% | 0.676 | 0.582 |
| Random Forest | Ensemble (Bagging) | 73.59% | 0.824 | 0.579 |
| XGBoost | Ensemble (Boosting) | 74.46% | 0.794 | 0.614 |
| LightGBM | Ensemble (Boosting) | 73.16% | 0.811 | 0.603 |
| CatBoost | Ensemble (Boosting) | 75.76% | 0.833 | 0.606 |
| **Hybrid RF-GBDT** | **Hybrid Ensemble** | **75.76%** | **0.834** | **0.627** |

The Hybrid RF-GBDT achieves the **best F1 score** and competitive accuracy, validating the effectiveness of combining bagging (RF) with boosting (GBDT).

---

## Project Structure

```
diabetes-risk-prediction/
│
├── app.py                          # Flask web application (all routes)
├── config.py                       # Configuration constants
├── requirements.txt                # Python dependencies
├── Procfile                        # Production server config (Gunicorn)
├── render.yaml                     # Render deployment config
│
├── data/
│   └── diabetes.csv                # PIMA Indians Diabetes Dataset (768 records)
│
├── ml/                             # Machine Learning Pipeline
│   ├── __init__.py
│   ├── dataset.py                  # Data loading & validation
│   ├── preprocessing.py            # Zero imputation + Min-Max normalization
│   ├── feature_extraction.py       # RF-based feature importance
│   ├── models.py                   # 7 model definitions + training functions
│   ├── evaluate.py                 # Metrics computation + plot generation
│   ├── train.py                    # Training pipeline orchestrator
│   └── predict.py                  # Real-time prediction for web app
│
├── saved_models/                   # Pre-trained model files
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── random_forest.joblib
│   ├── xgboost.joblib
│   ├── lightgbm.joblib
│   ├── catboost.joblib
│   ├── hybrid_rf_gbdt.joblib
│   ├── scaler.joblib               # Fitted MinMaxScaler
│   └── metrics_report.json         # Evaluation results
│
├── static/
│   ├── css/style.css               # Custom styling (blue-purple gradient theme)
│   ├── js/main.js                  # Charts, form validation, animations
│   └── plots/                      # Generated visualizations
│       ├── correlation_heatmap.png
│       ├── feature_importance.png
│       ├── confusion_matrices.png
│       ├── roc_curves.png
│       └── metrics_comparison.png
│
└── templates/                      # Jinja2 HTML templates
    ├── base.html                   # Base layout (navbar, footer)
    ├── index.html                  # Landing page
    ├── predict.html                # Prediction form + results
    ├── dashboard.html              # Model comparison dashboard
    ├── dataset.html                # Dataset info + EDA
    └── about.html                  # Research paper + team info
```

---

## Dataset

**PIMA Indians Diabetes Dataset** — from the National Institute of Diabetes and Digestive and Kidney Diseases.

| Property | Value |
|----------|-------|
| Records | 768 patients |
| Features | 8 clinical measurements |
| Target | Binary (0 = Non-diabetic, 1 = Diabetic) |
| Class Split | 500 Non-diabetic (65.1%) / 268 Diabetic (34.9%) |
| Source | NIDDK, Phoenix, Arizona |

**Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age

---

## ML Pipeline

```
diabetes.csv
    │
    ▼
[1] Replace biologically impossible zeros with column means
    │
    ▼
[2] Train-Test Split (70/30, stratified)
    │
    ▼
[3] Min-Max Normalization (fit on train, transform both)
    │
    ▼
[4] Feature Extraction (Random Forest importance)
    │
    ▼
[5] Train 7 Models (LR, DT, RF, XGBoost, LightGBM, CatBoost, Hybrid RF-GBDT)
    │
    ▼
[6] Evaluate (Accuracy, Precision, Recall, F1, AUC-ROC)
    │
    ▼
[7] Save models, scaler, metrics, plots
```

---

## Web Application

| Page | Description |
|------|-------------|
| **Home** (`/`) | Landing page with project overview and stats |
| **Predict** (`/predict`) | Enter 8 health parameters → get risk prediction from all 7 models |
| **Dashboard** (`/dashboard`) | Interactive Plotly charts comparing model performance |
| **Dataset** (`/dataset`) | Dataset description, feature info, correlation heatmap |
| **About** (`/about`) | Research paper summary, methodology, team info |
| **API** (`/api/predict`) | JSON endpoint for programmatic predictions |

---

## Installation & Setup

### Prerequisites
- Python 3.11+
- pip

### Steps

```bash
# Clone the repository
git clone https://github.com/xavierdsml/diabetes-risk-prediction.git
cd diabetes-risk-prediction

# Install dependencies
pip install -r requirements.txt

# (Optional) Retrain models from scratch
python -m ml.train

# Run the web application
python app.py
```

Open **http://localhost:5001** in your browser.

### Quick Test
```bash
# API prediction test
curl -X POST http://localhost:5001/api/predict \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies":6,"Glucose":148,"BloodPressure":72,"SkinThickness":35,"Insulin":0,"BMI":33.6,"DiabetesPedigreeFunction":0.627,"Age":50}'
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Pipeline | Python, scikit-learn, XGBoost, LightGBM, CatBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Plotly.js |
| Web Backend | Flask, Gunicorn |
| Web Frontend | HTML5, CSS3, Bootstrap 5, JavaScript |
| Model Storage | joblib |

---

## Key Features

- **7 ML Models** trained and compared with comprehensive metrics
- **Hybrid RF-GBDT** — novel two-stage ensemble (RF for feature extraction + GBDT for classification)
- **Real-time Prediction** — enter health parameters, get instant risk assessment
- **Interactive Dashboard** — Plotly.js charts with hover, zoom, and pan
- **Consensus Voting** — all 7 models vote, majority decides the final prediction
- **REST API** — JSON endpoint for integration with other systems
- **Responsive Design** — works on desktop, tablet, and mobile
- **Pre-trained Models** — no training needed, ready to predict out of the box

---

## Research

This project is based on the research paper:

**"A Hybrid GBDT Model for Advanced Diabetes Risk Prediction"**

*Authors:* Tushar Gupta, Manjari Gupta, Kunal Kaushik, Sanjana Jain

*Guide:* Gyanender Kumar (Assistant Professor, Dept. of CSE - Data Science, RKGIT)

*Institution:* Raj Kumar Goel Institute of Technology, Ghaziabad, U.P., India

---

## License

This project is developed for academic and research purposes.
