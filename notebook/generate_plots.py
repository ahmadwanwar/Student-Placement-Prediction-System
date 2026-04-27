"""
Exploratory Data Analysis & Model Visualisation
================================================

Generates a set of PNG plots into ``project/frontend/plots/`` so they can be
served by the Flask app and embedded in an Analysis tab in the UI.

Run AFTER train_model.py:
    python generate_plots.py
"""

from __future__ import annotations

import os
import pickle
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#4f46e5", "#dc2626"]  # Placed = indigo, Not Placed = red

# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, "data", "student_placement_prediction_dataset_2026.csv")
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
PLOTS_DIR = os.path.join(PROJECT_DIR, "frontend", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def save(fig: plt.Figure, name: str) -> str:
    path = os.path.join(PLOTS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path}")
    return path


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    if "student_id" in df.columns:
        df = df.drop(columns=["student_id"])

    target_col = "placement_status"
    salary_col = "salary_package_lpa"

    print("Generating plots...")

    # ------------------------------------------------------------------
    # 1. Target distribution
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df[target_col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, palette=PALETTE)
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts.values) * 0.01, f"{v:,}", ha="center", fontweight="bold")
    ax.set_title("Placement Status Distribution", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Number of Students")
    save(fig, "01_target_distribution.png")

    # ------------------------------------------------------------------
    # 2. Categorical features vs placement
    # ------------------------------------------------------------------
    cat_cols = ["gender", "branch", "college_tier", "volunteer_experience"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, col in zip(axes.flat, cat_cols):
        ct = (
            pd.crosstab(df[col], df[target_col], normalize="index")
            .reindex(columns=["Placed", "Not Placed"])
            * 100
        )
        ct.plot(kind="bar", stacked=True, ax=ax, color=PALETTE, edgecolor="white")
        ax.set_title(f"Placement Rate by {col.replace('_', ' ').title()}", fontweight="bold")
        ax.set_ylabel("% of students")
        ax.set_xlabel("")
        ax.set_ylim(0, 100)
        ax.legend(title="", loc="upper right", fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Categorical Features vs Placement", fontsize=14, fontweight="bold", y=1.00)
    save(fig, "02_categorical_vs_target.png")

    # ------------------------------------------------------------------
    # 3. Numeric distributions split by placement
    # ------------------------------------------------------------------
    num_cols_to_plot = [
        "cgpa",
        "coding_skill_score",
        "aptitude_score",
        "communication_skill_score",
        "logical_reasoning_score",
        "mock_interview_score",
        "attendance_percentage",
        "study_hours_per_day",
        "internships_count",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for ax, col in zip(axes.flat, num_cols_to_plot):
        for label, color in zip(["Placed", "Not Placed"], PALETTE):
            sns.kdeplot(
                df.loc[df[target_col] == label, col],
                ax=ax,
                color=color,
                fill=True,
                alpha=0.35,
                label=label,
                linewidth=1.5,
            )
        ax.set_title(col.replace("_", " ").title(), fontweight="bold", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.legend(fontsize=8)
    fig.suptitle("Numeric Feature Distributions by Placement", fontsize=14, fontweight="bold", y=1.00)
    save(fig, "03_numeric_distributions.png")

    # ------------------------------------------------------------------
    # 4. Correlation heatmap (numeric features + encoded target)
    # ------------------------------------------------------------------
    df_corr = df.copy()
    df_corr["placed"] = (df_corr[target_col] == "Placed").astype(int)
    numeric_for_corr = df_corr.select_dtypes(include=[np.number]).drop(
        columns=[salary_col], errors="ignore"
    )
    corr = numeric_for_corr.corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        annot_kws={"size": 7},
        square=True,
        cbar_kws={"shrink": 0.7},
        ax=ax,
    )
    ax.set_title("Correlation Heatmap (Numeric Features)", fontsize=13, fontweight="bold")
    save(fig, "04_correlation_heatmap.png")

    # ------------------------------------------------------------------
    # 5. Salary distribution among placed students
    # ------------------------------------------------------------------
    placed = df[df[target_col] == "Placed"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    sns.histplot(placed[salary_col], bins=30, kde=True, color="#4f46e5", ax=axes[0])
    axes[0].set_title("Salary Package Distribution (Placed Students)", fontweight="bold")
    axes[0].set_xlabel("Salary (LPA)")

    by_branch = placed.groupby("branch")[salary_col].mean().sort_values(ascending=False)
    sns.barplot(x=by_branch.index, y=by_branch.values, ax=axes[1], palette="viridis")
    for i, v in enumerate(by_branch.values):
        axes[1].text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9)
    axes[1].set_title("Average Salary by Branch", fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Avg Salary (LPA)")
    axes[1].tick_params(axis="x", rotation=20)
    save(fig, "05_salary_analysis.png")

    # ------------------------------------------------------------------
    # 6. CGPA buckets vs placement rate
    # ------------------------------------------------------------------
    df["cgpa_bucket"] = pd.cut(
        df["cgpa"],
        bins=[0, 6, 7, 8, 9, 10],
        labels=["<6", "6-7", "7-8", "8-9", "9-10"],
    )
    rate = df.groupby("cgpa_bucket")[target_col].apply(lambda s: (s == "Placed").mean() * 100)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(x=rate.index.astype(str), y=rate.values, ax=ax, palette="crest")
    for i, v in enumerate(rate.values):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    ax.set_title("Placement Rate by CGPA Bucket", fontsize=13, fontweight="bold")
    ax.set_xlabel("CGPA Range")
    ax.set_ylabel("% Placed")
    ax.set_ylim(0, max(rate.values) * 1.15)
    save(fig, "06_cgpa_vs_placement.png")

    # ------------------------------------------------------------------
    # 7. Boxplots: skill scores by placement status
    # ------------------------------------------------------------------
    score_cols = [
        "coding_skill_score",
        "communication_skill_score",
        "aptitude_score",
        "logical_reasoning_score",
        "mock_interview_score",
    ]
    long = df.melt(
        id_vars=[target_col], value_vars=score_cols, var_name="score", value_name="value"
    )
    long["score"] = long["score"].str.replace("_score", "").str.replace("_", " ").str.title()
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=long, x="score", y="value", hue=target_col, palette=PALETTE, ax=ax, fliersize=2
    )
    ax.set_title("Skill Scores by Placement Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.legend(title="")
    save(fig, "07_skill_boxplots.png")

    # ------------------------------------------------------------------
    # 8. Feature importance + 9. Confusion matrix (need trained model)
    # ------------------------------------------------------------------
    model_path = os.path.join(BACKEND_DIR, "model.pkl")
    meta_path = os.path.join(BACKEND_DIR, "meta.pkl")
    encoder_path = os.path.join(BACKEND_DIR, "encoder.pkl")
    scaler_path = os.path.join(BACKEND_DIR, "scaler.pkl")

    if all(os.path.exists(p) for p in (model_path, meta_path, encoder_path, scaler_path)):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        with open(encoder_path, "rb") as f:
            enc = pickle.load(f)
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        feature_columns = meta["feature_columns"]
        numeric_cols = meta["numeric_cols"]
        cat_cols = meta["cat_feature_cols"]
        target_classes = meta["target_classes"]

        # Feature importance
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values()
        fig, ax = plt.subplots(figsize=(9, 8))
        colors = sns.color_palette("crest", len(importances))
        ax.barh(importances.index, importances.values, color=colors)
        ax.set_title("Random Forest Feature Importance", fontsize=13, fontweight="bold")
        ax.set_xlabel("Importance")
        for i, v in enumerate(importances.values):
            ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=8)
        save(fig, "08_feature_importance.png")

        # Confusion matrix - rebuild test split with the same seed
        X = df[feature_columns].copy()
        y_raw = df[target_col].copy()
        for c in cat_cols:
            X[c] = enc["features"][c].transform(X[c].astype(str))
        X[numeric_cols] = scaler.transform(X[numeric_cols])
        y = enc["target"].transform(y_raw.astype(str))

        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=target_classes).plot(
            ax=ax, cmap="Blues", colorbar=False, values_format=","
        )
        ax.set_title("Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
        save(fig, "09_confusion_matrix.png")
    else:
        print("  Model artifacts not found - run train_model.py first to get plots 8 & 9.")

    print("\nDone. Plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
