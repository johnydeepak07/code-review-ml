# model/train.py
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')   # render to files only; needed on headless servers (Render/CI)
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBClassifier

# Anchor all paths to the project root so the script works from any
# working directory (this file lives in model/, so root is one level up).
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / 'data' / 'code_quality.csv'
OUTPUTS_DIR = ROOT / 'outputs'
MODEL_DIR = ROOT / 'model'

FEATURES = [
    'cyclomatic_complexity',
    'max_nesting_depth',
    'naming_entropy',
    'avg_function_length',
    'has_docstrings',
    'num_magic_numbers',
    'num_try_except'
]

def train():
    # --- 1. Load the dataset that build_dataset.py generated ---
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES]
    y = df['readable']

    # --- 2. Split into training set and test set ---
    # test_size=0.2 means 20% of rows are held back for evaluation
    # stratify=y means the split keeps the same readable/unreadable ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Train the XGBoost classifier ---
    clf = XGBClassifier(
        n_estimators=150,      # number of trees to build
        max_depth=4,           # how deep each tree can go
        learning_rate=0.1,     # how much each tree corrects the previous one
        eval_metric='logloss',
        random_state=42
    )
    clf.fit(X_train, y_train)

    # --- 4. Evaluate on the held-out test set ---
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=['hard_to_read', 'readable']))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_proba):.4f}")

    # --- 5. Cross-validation (more honest than a single split) ---
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring='roc_auc')
    print(f"5-Fold CV AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # --- 6. Save feature importance chart to outputs/ ---
    # The folder is gitignored, so it won't exist on a fresh clone (e.g. Render);
    # matplotlib does not create missing directories.
    OUTPUTS_DIR.mkdir(exist_ok=True)
    order = clf.feature_importances_.argsort()
    plt.figure(figsize=(8, 5))
    plt.barh([FEATURES[i] for i in order], clf.feature_importances_[order])
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance — Code Readability Classifier')
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / 'feature_importance.png', dpi=150)
    plt.close()
    print(f"Chart saved to {OUTPUTS_DIR / 'feature_importance.png'}")

    # --- 7. Save the trained model and feature list to model/ ---
    # joblib.dump serializes the Python object to a file
    # joblib.load (used in api/main.py) deserializes it back
    joblib.dump(clf, MODEL_DIR / 'readability_model.pkl')
    joblib.dump(FEATURES, MODEL_DIR / 'feature_names.pkl')
    print(f"Model saved: {MODEL_DIR / 'readability_model.pkl'}")
    print(f"Features saved: {MODEL_DIR / 'feature_names.pkl'}")

if __name__ == '__main__':
    train()
