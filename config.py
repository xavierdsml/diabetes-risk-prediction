import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')
METRICS_PATH = os.path.join(MODELS_DIR, 'metrics_report.json')

FEATURE_COLUMNS = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]
TARGET_COLUMN = 'Outcome'

ZERO_IMPUTE_COLUMNS = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

TEST_SIZE = 0.3
RANDOM_STATE = 42
