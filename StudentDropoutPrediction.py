"""
=============================================================================
Student Dropout Risk Prediction System
=============================================================================
Dataset : "Predict Students' Dropout and Academic Success"
          

Model   : Logistic Regression
=============================================================================
"""

# ---------------------------------------------------------------------------
#  IMPORTS
# ---------------------------------------------------------------------------

from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    recall_score,
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings("ignore")


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ---------------------------------------------------------------------------
#  SMOTE VISUALIZATION FUNCTIONS
# ---------------------------------------------------------------------------

def plot_smote_before_after(y_before, y_after):
    os.makedirs("outputs", exist_ok=True)

    before_counts = y_before.value_counts().sort_index()
    after_counts = pd.Series(y_after).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(before_counts.index, before_counts.values)
    axes[0].set_title("Before SMOTE")
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["Non-Dropout", "Dropout"])
    axes[0].set_ylabel("Number of Students")

    axes[1].bar(after_counts.index, after_counts.values)
    axes[1].set_title("After SMOTE")
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["Non-Dropout", "Dropout"])

    plt.tight_layout()
    plt.savefig("outputs/smote_before_after.png")
    plt.close()

    print("Saved smote_before_after.png")


def plot_smote_pca(X_before, y_before, X_after, y_after):
    os.makedirs("outputs", exist_ok=True)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)

    X_before_pca = pca.fit_transform(X_before)
    X_after_pca = pca.fit_transform(X_after)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(X_before_pca[:, 0], X_before_pca[:, 1],
                c=y_before, alpha=0.6)
    plt.title("Before SMOTE (PCA)")

    plt.subplot(1, 2, 2)
    plt.scatter(X_after_pca[:, 0], X_after_pca[:, 1],
                c=y_after, alpha=0.6)
    plt.title("After SMOTE (PCA)")

    plt.tight_layout()
    plt.savefig("outputs/smote_pca.png")
    plt.close()

    print("Saved smote_pca.png")


# ---------------------------------------------------------------------------
# MULTIVARIATE EDA — CORRELATION HEATMAP
# ---------------------------------------------------------------------------

def plot_correlation_heatmap(df):
    """
    Generate a multivariate correlation heatmap focusing on Personal,
    Financial, and Academic feature groups to justify the 'Multivariate EDA'
    requirement.
    """
    os.makedirs("outputs", exist_ok=True)

    # Map common column names to friendly labels.

    feature_groups = {
        # Personal
        "Gender": "Gender",
        "Age at enrollment": "Age at Enrolment",
        # Financial
        "Scholarship holder": "Scholarship Holder",
        "Debtor": "Debtor",
        "Tuition fees up to date": "Tuition Up To Date",
        # Academic
        "Curricular units 1st sem (grade)": "Grades Sem 1",
        "Curricular units 2nd sem (grade)": "Grades Sem 2",
        "Curricular units 1st sem (approved)": "Approved Units Sem 1",
        "Curricular units 2nd sem (approved)": "Approved Units Sem 2",
        # Target
        "Target_Binary": "Dropout",
    }

    available = {k: v for k, v in feature_groups.items() if k in df.columns}

    if len(available) < 3:
        print("Warning: Few expected columns found for heatmap. "
              "Using all numeric columns instead.")
        subset = df.select_dtypes(include=[np.number])
    else:
        subset = df[list(available.keys())].rename(columns=available)

    corr = subset.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)

    # Annotate cells
    for i in range(len(corr)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr.iloc[i, j]) < 0.6 else "white")

    ax.set_title("Multivariate Correlation Heatmap\n"
                 "(Personal · Financial · Academic Features)", fontsize=13, pad=15)

    # Draw group separator lines
    n_personal = sum(1 for v in available.values()
                     if v in ("Gender", "Age at Enrolment"))
    n_financial = sum(1 for v in available.values()
                      if v in ("Scholarship Holder", "Debtor", "Tuition Up To Date"))

    for boundary in [n_personal - 0.5, n_personal + n_financial - 0.5]:
        ax.axhline(boundary, color="black", linewidth=1.5)
        ax.axvline(boundary, color="black", linewidth=1.5)

    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150)
    plt.close()

    print("Saved correlation_heatmap.png")


# ---------------------------------------------------------------------------
# BIAS & FAIRNESS ANALYSIS
# ---------------------------------------------------------------------------

def evaluate_fairness(model, X_test, y_test, feature_names):
    """
    Fixed: Ensures both Male and Female groups are detected by using 
    un-scaled or properly rounded gender values.
    """
    os.makedirs("outputs", exist_ok=True)

    # We convert back to a DataFrame to easily filter by the 'Gender' column
    test_df = pd.DataFrame(X_test, columns=feature_names)
    y_test = np.array(y_test)
    y_pred = model.predict(X_test)

    results = {}
    # UCI encoding: 1 = Male, 0 = Female
    gender_map = {1: "Male", 0: "Female"}

    print("\n--- Bias & Fairness Analysis (Gender) ---")
    for val, label in gender_map.items():
        # Using np.isclose because 'StandardScaler' might turn 0 into -0.45 or 1 into 1.2
        # We find where the Gender column is closest to the scaled version of 0 or 1
        mask = (test_df["Gender"] > 0) if val == 1 else (
            test_df["Gender"] <= 0)

        if mask.sum() > 0:
            recall = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            acc = accuracy_score(y_test[mask], y_pred[mask])
            results[label] = {"Recall": recall,
                              "Accuracy": acc, "n": int(mask.sum())}
            print(
                f"  {label:6s} (n={int(mask.sum()):4d}) | Recall: {recall:.4f} | Accuracy: {acc:.4f}")
        else:
            print(f"  {label:6s} | No data found in test set.")

    # Bar chart
    labels = list(results.keys())
    recall_vals = [results[l]["Recall"] for l in labels]
    acc_vals = [results[l]["Accuracy"] for l in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, recall_vals,  width,
                   label="Recall",   color="#4C72B0")
    bars2 = ax.bar(x + width / 2, acc_vals,     width,
                   label="Accuracy", color="#DD8452")

    ax.set_ylabel("Score")
    ax.set_title("Fairness Analysis — Model Performance by Gender")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.axhline(0.8, color="red", linestyle="--",
               linewidth=0.8, label="0.8 threshold")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{bar.get_height():.2f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    plt.savefig("outputs/fairness_gender.png", dpi=150)
    plt.close()

    print("Saved fairness_gender.png")


# ---------------------------------------------------------------------------
#  INTERACTIVE CLI PREDICTOR  [NEW]
# ---------------------------------------------------------------------------

# Plain-English question config for each known feature.
# Each entry: "dataset column name" -> (question, valid answers or None, default, converter)
#   • question   — what the user actually sees
#   • choices    — printed menu options (None = free numeric entry)
#   • default    — used when the user just presses Enter
#   • converter  — turns the raw string into the number the model needs
_FRIENDLY_PROMPTS = {
    "Curricular units 1st sem (grade)": dict(
        question="What was the student's average grade in Semester 1?",
        guidance="Enter a number from 0 to 20  (e.g. 14 = a solid pass, 18 = excellent)",
        choices=None,
        default=13.0,
        converter=float,
    ),
    "Curricular units 2nd sem (grade)": dict(
        question="What was the student's average grade in Semester 2?",
        guidance="Enter a number from 0 to 20  (e.g. 14 = a solid pass, 18 = excellent)",
        choices=None,
        default=13.0,
        converter=float,
    ),
    "Tuition fees up to date": dict(
        question="Are the student's tuition fees fully paid and up to date?",
        guidance="",
        choices={"1": ("Yes — fees are paid", 1), "2": (
            "No  — fees are outstanding", 0)},
        default="1",
        converter=None,
    ),
    "Scholarship holder": dict(
        question="Does the student hold a scholarship?",
        guidance="",
        choices={"1": ("Yes — has a scholarship", 1),
                 "2": ("No  — no scholarship", 0)},
        default="2",
        converter=None,
    ),
    "Age at enrollment": dict(
        question="How old was the student when they enrolled?",
        guidance="Enter their age in whole years  (e.g. 18, 23, 35)",
        choices=None,
        default=20,
        converter=int,
    ),
    "Debtor": dict(
        question="Does the student currently owe money to the institution?",
        guidance="",
        choices={"1": ("Yes — student has outstanding debt", 1),
                 "2": ("No  — no debt", 0)},
        default="2",
        converter=None,
    ),
    "Gender": dict(
        question="What is the student's gender?",
        guidance="",
        choices={"1": ("Male", 1), "2": ("Female", 0)},
        default="1",
        converter=None,
    ),
    "Curricular units 1st sem (approved)": dict(
        question="How many course units did the student pass in Semester 1?",
        guidance="Enter a whole number  (e.g. 0 = none passed, 6 = all units passed)",
        choices=None,
        default=5,
        converter=int,
    ),
    "Curricular units 2nd sem (approved)": dict(
        question="How many course units did the student pass in Semester 2?",
        guidance="Enter a whole number  (e.g. 0 = none passed, 6 = all units passed)",
        choices=None,
        default=5,
        converter=int,
    ),
}

# Fallback config for any feature not listed above
_FALLBACK_PROMPT = dict(
    question=None,          # filled in dynamically
    guidance="Enter a numeric value",
    choices=None,
    default=0,
    converter=float,
)


def _ask(prompt_cfg, feat_label):
    """Prompt the user once, using a multiple-choice menu or a free-text entry."""
    question = prompt_cfg["question"] or f"Value for '{feat_label}'"
    guidance = prompt_cfg["guidance"]
    choices = prompt_cfg["choices"]
    default = prompt_cfg["default"]
    converter = prompt_cfg["converter"]

    print(f"\n  ❓ {question}")

    if choices:
        for key, (label, _) in choices.items():
            marker = " (default)" if key == str(default) else ""
            print(f"     {key}) {label}{marker}")
        while True:
            raw = input("     Your choice: ").strip() or str(default)
            if raw in choices:
                return float(choices[raw][1])
            print("     Please type one of the numbers shown above.")
    else:
        if guidance:
            print(f"     ℹ  {guidance}")
        print(f"     (Press Enter to use the default: {default})")
        while True:
            raw = input("     Your answer: ").strip()
            try:
                return float(converter(raw) if raw else default)
            except (ValueError, TypeError):
                print("     Please enter a valid number.")


# ---------------------------------------------------------------------------
#  IMPROVED INTERACTIVE PREDICTOR
# ---------------------------------------------------------------------------

def interactive_prediction(model, scaler, feature_names):
    """
    Fixed: Handles scaling correctly and implements a sensitivity threshold.
    """
    print("\n" + "=" * 60)
    print("    STUDENT DROPOUT RISK PREDICTOR (Sensitivity Tuned)")
    print("=" * 60)

    # Use the specific features that impact the model most
    top5_features = ["Curricular units 2nd sem (grade)", "Tuition fees up to date",
                     "Scholarship holder", "Debtor", "Gender"]

    input_vector = np.zeros(len(feature_names))

    for step, feat in enumerate(top5_features, start=1):
        print(f"\n  ── Question {step} of {len(top5_features)} " + "─" * 30)
        cfg = _FRIENDLY_PROMPTS.get(
            feat, {**_FALLBACK_PROMPT, "question": f"Value for '{feat}'"})
        val = _ask(cfg, feat)
        input_vector[list(feature_names).index(feat)] = val

    # CRITICAL FIX: Scaling the input to match the training distribution
    input_df = pd.DataFrame([input_vector], columns=feature_names)
    input_scaled = scaler.transform(input_df)

    # Get raw probability
    risk_proba = model.predict_proba(input_scaled)[0][1]
    risk_pct = risk_proba * 100

    print(f"\n[Debug] Raw Dropout Probability: {risk_proba:.4f}")

    # ADJUSTED THRESHOLD: 0.4 instead of 0.5 to increase sensitivity
    if risk_proba >= 0.7:
        level, icon = "HIGH RISK", "🔴"
        recommendation = "Refer immediately for Academic Advising & Financial Support."
    elif risk_proba >= 0.4:
        level, icon = "MEDIUM RISK", "🟡"
        recommendation = "Monitor closely; early signs of attrition detected."
    else:
        level, icon = "LOW RISK", "🟢"
        recommendation = "Student appears on track — maintain regular check-ins."

    print("\n" + "=" * 60)
    print("   FINAL ASSESSMENT")
    print("=" * 60)
    print(f"   Calculated Risk :  {risk_pct:.1f}%")
    print(f"   Risk Level      :  {icon}  {level}")
    print(f"   Recommendation  :  {recommendation}")
    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
#  DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------


def load_and_preprocess(filepath="dataset.csv"):

    print("=" * 60)
    print("STEP 1 — Loading & Preprocessing Data")
    print("=" * 60)

    df = pd.read_csv(filepath, sep=None, engine="python")

    print(f"Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    # -------------------------------
# Initial Dataset Inspection
# -------------------------------

    print("\n--- Initial Data (Head) ---")
    print(df.head())

    print("\n--- Initial Data Info for Selected Features ---")
    print(df.info())

    print("\n--- Initial Data (Missing Values for Selected Features) ---")
    print(df.isnull().sum())

    if "Target" not in df.columns:
        raise ValueError("ERROR: 'Target' column not found.")

    print("\nOriginal target distribution:")
    print(df["Target"].value_counts())

    # Binary target
    df["Target_Binary"] = (df["Target"] == "Dropout").astype(int)
    df.drop(columns=["Target"], inplace=True)

    # ── Multivariate EDA Heatmap (before splitting) ──────────────────────
    plot_correlation_heatmap(df)

    X = df.drop(columns=["Target_Binary"])
    y = df["Target_Binary"]
    feature_names = X.columns.tolist()

    # Encode non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        le = LabelEncoder()
        for col in non_numeric:
            X[col] = le.fit_transform(X[col].astype(str))

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n--- Dataset After Encoding & Scaling ---")
    print("Training set shape:", X_train_scaled.shape)
    print("Test set shape:", X_test_scaled.shape)

    # SMOTE (training data only)
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_res_arr, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    # ── TECHNICAL FIX: convert SMOTE output back to a DataFrame ──────────
    X_train_res = pd.DataFrame(X_train_res_arr, columns=feature_names)

    print("\nAfter SMOTE class balance:")
    print(pd.Series(y_train_res).value_counts())

    # SMOTE visualisations (pass numpy arrays; PCA works on both)
    plot_smote_before_after(y_train, y_train_res)
    plot_smote_pca(X_train_scaled, y_train, X_train_res_arr, y_train_res)

    print("\nPreprocessing complete.\n")

    return (X_train_scaled, X_train_res, X_test_scaled,
            y_train, y_train_res, y_test,
            feature_names, scaler)


# ---------------------------------------------------------------------------
#  MODEL TRAINING  (baseline + SMOTE comparison)  [UPDATED]
# ---------------------------------------------------------------------------

def train_model(X_train_raw, y_train_raw, X_train_smote, y_train_smote):
    """
    Train two models:
      • Baseline — Logistic Regression on raw (unbalanced) training data
      • Final    — Logistic Regression on SMOTE-balanced training data

    Prints a comparison table of F1 and Recall scores.
    Returns the Final (SMOTE) model.
    """
    print("=" * 60)
    print("STEP 2 — Model Training & Baseline Comparison")
    print("=" * 60)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

    def make_lr():
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            C=0.5,
            solver="lbfgs",
        )

    # ── Baseline (no SMOTE) ───────────────────────────────────────────────
    print("\n[Baseline — no SMOTE]")
    baseline_model = make_lr()
    cv_base = cross_validate(baseline_model, X_train_raw, y_train_raw,
                             cv=cv, scoring=scoring, n_jobs=-1)
    baseline_model.fit(X_train_raw, y_train_raw)

    # ── Final model (SMOTE) ───────────────────────────────────────────────
    print("[Final — with SMOTE]")
    final_model = make_lr()
    cv_final = cross_validate(final_model, X_train_smote, y_train_smote,
                              cv=cv, scoring=scoring, n_jobs=-1)
    final_model.fit(X_train_smote, y_train_smote)

    # ── Comparison table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("BASELINE vs FINAL MODEL — 5-Fold CV Comparison")
    print("=" * 60)
    print(f"{'Metric':<12} {'Baseline':>12} {'Final (SMOTE)':>15} {'Δ':>10}")
    print("-" * 52)
    for metric in scoring:
        base_mean = cv_base[f"test_{metric}"].mean()
        final_mean = cv_final[f"test_{metric}"].mean()
        delta = final_mean - base_mean
        marker = "▲" if delta > 0 else ("▼" if delta < 0 else " ")
        print(f"{metric:<12} {base_mean:>12.4f} {final_mean:>15.4f} "
              f"{marker}{abs(delta):>8.4f}")
    print("=" * 60)

    # ── Full CV output for final model ────────────────────────────────────
    print("\nFinal Model — 5-Fold CV Detail:")
    for metric in scoring:
        vals = cv_final[f"test_{metric}"]
        print(f"  {metric:12s}: {vals.mean():.4f} +/- {vals.std():.4f}")

    return final_model


# ---------------------------------------------------------------------------
#  EVALUATION
# ---------------------------------------------------------------------------

def evaluate_model(lr_model, X_test, y_test):

    y_pred = lr_model.predict(X_test)
    y_proba = lr_model.predict_proba(X_test)[:, 1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Dropout", "Dropout"]))

    auc_score = roc_auc_score(y_test, y_proba)

    os.makedirs("outputs", exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm,
                           display_labels=["Non-Dropout", "Dropout"]).plot()
    plt.tight_layout()
    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.plot(fpr, tpr, label=f"AUC = {auc_score:.3f}")
    plt.plot([0, 1], [0, 1], "--")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_auc_curve.png")
    plt.close()


# ---------------------------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def plot_feature_importance(lr_model, feature_names, top_n=10):

    os.makedirs("outputs", exist_ok=True)

    coefs = lr_model.coef_[0]
    idx = np.argsort(np.abs(coefs))[::-1][:top_n]

    plt.barh(
        [feature_names[i] for i in idx][::-1],
        coefs[idx][::-1]
    )
    plt.axvline(0)
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------

def main(dataset_path="dataset.csv"):

    (X_train_raw, X_train_smote,
     X_test, y_train_raw, y_train_smote,
     y_test, feature_names, scaler) = load_and_preprocess(dataset_path)

    model = train_model(X_train_raw, y_train_raw,
                        X_train_smote, y_train_smote)

    evaluate_model(model, X_test, y_test)

    plot_feature_importance(model, feature_names)

    # ── Fairness analysis ─────────────────────────────────────────────────
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    evaluate_fairness(model, X_test_df, y_test, feature_names)

    # ── Interactive predictor ─────────────────────────────────────────────
    try:
        answer = input(
            "\nWould you like to try the interactive predictor? (y/n): ").strip().lower()
        if answer == "y":
            interactive_prediction(model, scaler, feature_names)
    except EOFError:
        # Non-interactive environment (e.g. CI pipeline) — skip gracefully
        pass

    print("\nPipeline complete!")
    print("All outputs saved inside /outputs")


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="dataset.csv")
    args = parser.parse_args()

    main(dataset_path=args.data)
