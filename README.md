# Student Placement Prediction

End-to-end machine learning web application that predicts whether a student
will be placed, based on academic, skill, and engagement features.

- **Model**: Random Forest Classifier (scikit-learn)
- **Backend**: Flask REST API
- **Frontend**: Plain HTML + CSS + JavaScript (no build step)
- **Dataset**: `student_placement_prediction_dataset_2026.csv` (100,000 rows)

## Project Structure

```
project/
├── backend/
│   ├── app.py              # Flask API + serves the frontend
│   ├── model.pkl           # Trained Random Forest (generated)
│   ├── scaler.pkl          # StandardScaler (generated)
│   ├── encoder.pkl         # LabelEncoders (generated)
│   ├── meta.pkl            # Feature/target metadata (generated)
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── notebook/
│   ├── train_model.py      # Standalone training script
│   ├── generate_plots.py   # EDA + model visualisations
│   └── training.ipynb      # Same pipeline as a Jupyter notebook
├── data/
│   └── student_placement_prediction_dataset_2026.csv
└── README.md
```

## How to Run on Windows (Python 3.11)

Open **Command Prompt** or **PowerShell** in the `project/` folder.

### 1. Create and activate a virtual environment

```cmd
py -3.11 -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies

```cmd
pip install -r backend\requirements.txt
```

### 3. Train the model

This generates `model.pkl`, `scaler.pkl`, `encoder.pkl`, and `meta.pkl`
inside `backend/`. Run it once before starting the server.

```cmd
python notebook\train_model.py
```

You should see the data summary, evaluation metrics (accuracy, precision,
recall, F1, confusion matrix), and the saved file paths.

### 4. Generate the EDA + model visualisations

This writes 9 PNG charts into `frontend/plots/` (target distribution,
categorical breakdowns, KDEs, correlation heatmap, salary analysis, CGPA
buckets, skill boxplots, feature importance, confusion matrix). They are
served by Flask and shown in the **Analysis** tab of the UI.

```cmd
python notebook\generate_plots.py
```

### 5. Start the Flask server

```cmd
cd backend
python app.py
```

The server starts on **http://localhost:5000**.

### 6. Open the app

Open <http://localhost:5000> in a browser. The **Predict** tab has a form
built dynamically from the trained model's feature schema - click **Fill
sample** to populate example values and then **Predict**. The **Analysis**
tab shows the 9 EDA charts.

## API

### `GET /schema`

Returns the feature metadata used by the frontend.

### `POST /predict`

**Request body** — a JSON object with all 23 features:

```json
{
  "age": 21,
  "gender": "Male",
  "cgpa": 8.2,
  "branch": "CSE",
  "college_tier": "Tier 1",
  "internships_count": 2,
  "projects_count": 4,
  "certifications_count": 3,
  "coding_skill_score": 85,
  "aptitude_score": 78,
  "communication_skill_score": 80,
  "logical_reasoning_score": 82,
  "hackathons_participated": 2,
  "github_repos": 6,
  "linkedin_connections": 500,
  "mock_interview_score": 78,
  "attendance_percentage": 90,
  "backlogs": 0,
  "extracurricular_score": 70,
  "leadership_score": 65,
  "volunteer_experience": "Yes",
  "sleep_hours": 7,
  "study_hours_per_day": 4
}
```

**Response**:

```json
{
  "prediction": "Placed",
  "probability": 0.68,
  "class_probabilities": { "Not Placed": 0.32, "Placed": 0.68 }
}
```

**Errors** return HTTP 400 with a clear message, e.g.
`{"error": "Missing fields: ['cgpa', ...]"}`.

## Notes

- Random Forest is robust to feature scaling, captures non-linear feature
  interactions, and rarely overfits with sensible hyperparameters — a strong
  default for tabular classification problems like placement prediction.
- The same `StandardScaler` and `LabelEncoder` objects fit during training
  are loaded by the Flask app so prediction-time preprocessing exactly
  matches training-time preprocessing.
- The Flask app also serves the static frontend, so a single `python app.py`
  command runs both the API and the UI.
