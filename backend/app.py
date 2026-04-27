"""
Flask backend for the Student Placement Prediction app.

Endpoints:
    GET  /              -> health/info
    GET  /schema        -> feature schema (used by the frontend to build the form)
    POST /predict       -> JSON in, JSON out: {prediction, probability}

Run:
    python app.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
FRONTEND_DIR = os.path.join(PROJECT_DIR, "frontend")

MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "encoder.pkl")
META_PATH = os.path.join(BASE_DIR, "meta.pkl")

# ----------------------------------------------------------------------
# Load artifacts once at startup
# ----------------------------------------------------------------------
for p in (MODEL_PATH, SCALER_PATH, ENCODER_PATH, META_PATH):
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"Missing artifact: {p}\n"
            "Train the model first:  python notebook/train_model.py"
        )

with open(MODEL_PATH, "rb") as f:
    MODEL = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    SCALER = pickle.load(f)
with open(ENCODER_PATH, "rb") as f:
    _enc = pickle.load(f)
    FEATURE_ENCODERS = _enc["features"]
    TARGET_ENCODER = _enc["target"]
with open(META_PATH, "rb") as f:
    META = pickle.load(f)

FEATURE_COLUMNS: list[str] = META["feature_columns"]
NUMERIC_COLS: list[str] = META["numeric_cols"]
CAT_FEATURE_COLS: list[str] = META["cat_feature_cols"]
TARGET_CLASSES: list[str] = META["target_classes"]
CATEGORICAL_OPTIONS: dict[str, list[str]] = META["categorical_options"]

# ----------------------------------------------------------------------
# Validation rules for numeric fields
# ----------------------------------------------------------------------
# Each rule:  {"min": float, "max": float, "integer": bool}
# - "integer": True  -> value must be a whole number
# - The frontend reads this from /schema and applies the same bounds
#   client-side; the backend always re-validates on /predict.
VALIDATION_RULES: dict[str, dict] = {
    "age":                       {"min": 1,  "max": 80,    "integer": True},
    "cgpa":                      {"min": 0,  "max": 10,    "integer": False},
    "internships_count":         {"min": 0,  "max": 50,    "integer": True},
    "projects_count":            {"min": 0,  "max": 50,    "integer": True},
    "certifications_count":      {"min": 0,  "max": 50,    "integer": True},
    "coding_skill_score":        {"min": 0,  "max": 100,   "integer": False},
    "aptitude_score":            {"min": 0,  "max": 100,   "integer": False},
    "communication_skill_score": {"min": 0,  "max": 100,   "integer": False},
    "logical_reasoning_score":   {"min": 0,  "max": 100,   "integer": False},
    "hackathons_participated":   {"min": 0,  "max": 50,    "integer": True},
    "github_repos":              {"min": 0,  "max": 500,   "integer": True},
    "linkedin_connections":      {"min": 0,  "max": 30000, "integer": True},
    "mock_interview_score":      {"min": 0,  "max": 100,   "integer": False},
    "attendance_percentage":     {"min": 0,  "max": 100,   "integer": False},
    "backlogs":                  {"min": 0,  "max": 30,    "integer": True},
    "extracurricular_score":     {"min": 0,  "max": 100,   "integer": False},
    "leadership_score":          {"min": 0,  "max": 100,   "integer": False},
    "sleep_hours":               {"min": 0,  "max": 24,    "integer": False},
    "study_hours_per_day":       {"min": 0,  "max": 24,    "integer": False},
}

# ----------------------------------------------------------------------
# Flask app - also serves the static frontend so a single command runs everything
# ----------------------------------------------------------------------
app = Flask(__name__, static_folder=FRONTEND_DIR, static_url_path="")
CORS(app)


@app.route("/")
def index():
    """Serve the frontend index page."""
    if os.path.exists(os.path.join(FRONTEND_DIR, "index.html")):
        return send_from_directory(FRONTEND_DIR, "index.html")
    return jsonify({"status": "ok", "message": "Student Placement Prediction API"})


@app.route("/schema", methods=["GET"])
def schema():
    """Return feature metadata so the frontend can build its form dynamically."""
    return jsonify(
        {
            "feature_columns": FEATURE_COLUMNS,
            "numeric_cols": NUMERIC_COLS,
            "categorical_cols": CAT_FEATURE_COLS,
            "categorical_options": CATEGORICAL_OPTIONS,
            "target_classes": TARGET_CLASSES,
            "validation_rules": VALIDATION_RULES,
        }
    )


def _validate_and_build_row(
    payload: dict,
) -> tuple[pd.DataFrame | None, list[str] | None]:
    """Validate the incoming JSON and return a 1-row DataFrame ready for the pipeline.

    Returns either (DataFrame, None) or (None, list_of_error_messages).
    """
    if not isinstance(payload, dict):
        return None, ["Request body must be a JSON object."]

    missing = [c for c in FEATURE_COLUMNS if c not in payload]
    if missing:
        return None, [f"Missing fields: {missing}"]

    errors: list[str] = []
    row: dict = {}

    for col in FEATURE_COLUMNS:
        value = payload[col]

        if col in CAT_FEATURE_COLS:
            value = str(value)
            allowed = CATEGORICAL_OPTIONS.get(col, [])
            if allowed and value not in allowed:
                errors.append(
                    f"'{col}': must be one of {allowed} (got '{value}')."
                )
                continue
            row[col] = value
            continue

        # Numeric field
        try:
            num = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{col}': must be a number (got '{value}').")
            continue

        if num != num:  # NaN check
            errors.append(f"'{col}': must be a finite number.")
            continue

        rule = VALIDATION_RULES.get(col)
        if rule is not None:
            lo, hi = rule["min"], rule["max"]
            if num < lo or num > hi:
                errors.append(f"'{col}': must be between {lo} and {hi} (got {num}).")
                continue
            if rule.get("integer") and float(num).is_integer() is False:
                errors.append(f"'{col}': must be a whole number (got {num}).")
                continue

        row[col] = num

    if errors:
        return None, errors

    return pd.DataFrame([row], columns=FEATURE_COLUMNS), None


@app.route("/predict", methods=["POST"])
def predict():
    """Predict placement status for a single student."""
    try:
        payload = request.get_json(force=True, silent=True)
        df, errs = _validate_and_build_row(payload)
        if errs:
            return jsonify({"error": "Invalid input.", "details": errs}), 400

        # Apply the same encoders used during training.
        for col in CAT_FEATURE_COLS:
            le = FEATURE_ENCODERS[col]
            df[col] = le.transform(df[col].astype(str))

        # Apply the same scaler to the numeric columns.
        df[NUMERIC_COLS] = SCALER.transform(df[NUMERIC_COLS])

        proba = MODEL.predict_proba(df)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = TARGET_ENCODER.inverse_transform([pred_idx])[0]

        return jsonify(
            {
                "prediction": str(pred_label),
                "probability": round(float(proba[pred_idx]), 4),
                "class_probabilities": {
                    str(TARGET_CLASSES[i]): round(float(proba[i]), 4)
                    for i in range(len(TARGET_CLASSES))
                },
            }
        )

    except Exception as exc:  # noqa: BLE001 - return clean JSON for any failure
        return jsonify({"error": f"Internal error: {exc}"}), 500


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
