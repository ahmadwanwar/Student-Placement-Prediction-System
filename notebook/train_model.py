"""
Student Placement Prediction - Model Training Script
=====================================================

This script performs end-to-end training:
1. Loads and analyzes the dataset
2. Preprocesses features (handles missing values, encodes categoricals, scales numerics)
3. Trains a Random Forest classifier
4. Evaluates the model
5. Saves model + preprocessing objects so the Flask backend can reuse them

Run:
    python train_model.py
"""

import os
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "student_placement_prediction_dataset_2026.csv")
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
os.makedirs(BACKEND_DIR, exist_ok=True)

MODEL_PATH = os.path.join(BACKEND_DIR, "model.pkl")
SCALER_PATH = os.path.join(BACKEND_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BACKEND_DIR, "encoder.pkl")
META_PATH = os.path.join(BACKEND_DIR, "meta.pkl")


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Data Analysis
    # ------------------------------------------------------------------
    print("=" * 70)
    print("STEP 1: DATA ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(DATA_PATH)

    print("\nShape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values per column:")
    print(df.isnull().sum())

    # Drop the identifier column - it carries no predictive information.
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    # Identify categorical columns (object dtype).
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    print("\nCategorical columns:", categorical_cols)
    for col in categorical_cols:
        print(f"  {col}: {df[col].unique().tolist()}")

    # ------------------------------------------------------------------
    # 2. Target / Feature split
    # ------------------------------------------------------------------
    # The dataset has two potential targets:
    #   - placement_status  (Placed / Not Placed)   -> classification target
    #   - salary_package_lpa (numeric, 0 if not placed) -> not used here
    target_col = "placement_status"
    drop_cols = [target_col, "salary_package_lpa"]

    y_raw = df[target_col].copy()
    X = df.drop(columns=drop_cols)

    # ------------------------------------------------------------------
    # 3. Missing value handling
    # ------------------------------------------------------------------
    # Numeric -> fill with median (robust to outliers).
    # Categorical -> fill with mode.
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_feature_cols = [c for c in categorical_cols if c in X.columns]

    for col in numeric_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
    for col in cat_feature_cols:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].mode().iloc[0])

    # ------------------------------------------------------------------
    # 4. Encoding
    # ------------------------------------------------------------------
    # Use LabelEncoder per categorical column. We store the fitted encoders
    # in a dict so the Flask service can apply the same mapping at predict time.
    feature_encoders: dict[str, LabelEncoder] = {}
    for col in cat_feature_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        feature_encoders[col] = le

    # Encode the target.
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y_raw.astype(str))
    print("\nTarget classes (encoded order):", list(target_encoder.classes_))

    # ------------------------------------------------------------------
    # 5. Scaling
    # ------------------------------------------------------------------
    # Standard-scale only the originally numeric columns.
    feature_columns = X.columns.tolist()
    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    print("\nProcessed feature matrix shape:", X.shape)

    # ------------------------------------------------------------------
    # 6. Model Training
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2: MODEL BUILDING")
    print("=" * 70)
    # Random Forest is robust to feature scaling, handles non-linear interactions
    # well, and rarely overfits with sensible hyperparameters - a strong default
    # for tabular classification problems like placement prediction.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=4,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # ------------------------------------------------------------------
    # 7. Evaluation
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)

    print("\nAccuracy :", round(accuracy_score(y_test, y_pred), 4))
    print("Precision:", round(precision_score(y_test, y_pred, average="weighted"), 4))
    print("Recall   :", round(recall_score(y_test, y_pred, average="weighted"), 4))
    print("F1-score :", round(f1_score(y_test, y_pred, average="weighted"), 4))
    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

    # ------------------------------------------------------------------
    # 8. Persist artifacts for the Flask service
    # ------------------------------------------------------------------
    meta = {
        "feature_columns": feature_columns,
        "numeric_cols": numeric_cols,
        "cat_feature_cols": cat_feature_cols,
        "target_classes": list(target_encoder.classes_),
        "categorical_options": {
            col: list(map(str, feature_encoders[col].classes_)) for col in cat_feature_cols
        },
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(ENCODER_PATH, "wb") as f:
        pickle.dump({"features": feature_encoders, "target": target_encoder}, f)
    with open(META_PATH, "wb") as f:
        pickle.dump(meta, f)

    print("\nSaved:")
    print(f"  {MODEL_PATH}")
    print(f"  {SCALER_PATH}")
    print(f"  {ENCODER_PATH}")
    print(f"  {META_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
