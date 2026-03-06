"""
train_model.py — AI Health Risk Prediction Model Trainer
Generates synthetic patient data and trains Random Forest classifiers
for Diabetes, Heart Disease, and Kidney Disease prediction.
Run this script ONCE before starting the Flask app.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import pickle
import os

# ─── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(42)

N = 2000  # number of synthetic patients


# ─── Helper: generate correlated risk label ──────────────────────────────────
def compute_risk(score):
    """Convert a continuous risk score to 0=Low, 1=Medium, 2=High."""
    if score < 0.33:
        return 0
    elif score < 0.66:
        return 1
    else:
        return 2


# ════════════════════════════════════════════════════════════════════════════
# 1. GENERATE SYNTHETIC DATASET
# ════════════════════════════════════════════════════════════════════════════
def generate_dataset(n=N):
    """
    Produce realistic-looking synthetic patient records.
    Each feature's range is grounded in medical reference values.
    """
    age = np.random.randint(20, 80, n)
    bmi = np.round(np.random.uniform(17, 42, n), 1)
    blood_pressure = np.random.randint(70, 180, n)      # systolic mmHg
    glucose = np.random.randint(70, 300, n)              # mg/dL fasting
    cholesterol = np.random.randint(120, 320, n)         # mg/dL total
    smoking = np.random.randint(0, 2, n)                 # 0=No 1=Yes
    physical_activity = np.random.randint(0, 8, n)       # days/week
    family_history = np.random.randint(0, 2, n)          # 0=No 1=Yes
    hba1c = np.round(np.random.uniform(4.5, 12.0, n), 1)
    creatinine = np.round(np.random.uniform(0.5, 5.0, n), 2)

    df = pd.DataFrame({
        'age': age,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'glucose': glucose,
        'cholesterol': cholesterol,
        'smoking': smoking,
        'physical_activity': physical_activity,
        'family_history': family_history,
        'hba1c': hba1c,
        'creatinine': creatinine,
    })

    # ── Diabetes risk: glucose + HbA1c dominant ──────────────────────────
    d_score = (
        (df['glucose'] - 70) / 230 * 0.35 +
        (df['hba1c'] - 4.5) / 7.5 * 0.30 +
        (df['bmi'] - 17) / 25 * 0.15 +
        (df['age'] - 20) / 60 * 0.10 +
        df['family_history'] * 0.10
    ) + np.random.normal(0, 0.05, n)
    d_score = np.clip(d_score, 0, 1)
    df['diabetes_risk'] = [compute_risk(s) for s in d_score]

    # ── Heart disease risk: BP + cholesterol + smoking dominant ──────────
    h_score = (
        (df['blood_pressure'] - 70) / 110 * 0.30 +
        (df['cholesterol'] - 120) / 200 * 0.25 +
        df['smoking'] * 0.20 +
        (df['age'] - 20) / 60 * 0.15 +
        (1 - df['physical_activity'] / 7) * 0.10
    ) + np.random.normal(0, 0.05, n)
    h_score = np.clip(h_score, 0, 1)
    df['heart_risk'] = [compute_risk(s) for s in h_score]

    # ── Kidney disease risk: creatinine + BP + glucose dominant ─────────
    k_score = (
        (df['creatinine'] - 0.5) / 4.5 * 0.35 +
        (df['blood_pressure'] - 70) / 110 * 0.25 +
        (df['glucose'] - 70) / 230 * 0.20 +
        (df['age'] - 20) / 60 * 0.10 +
        df['smoking'] * 0.10
    ) + np.random.normal(0, 0.05, n)
    k_score = np.clip(k_score, 0, 1)
    df['kidney_risk'] = [compute_risk(s) for s in k_score]

    return df


# ════════════════════════════════════════════════════════════════════════════
# 2. TRAIN & PERSIST MODELS
# ════════════════════════════════════════════════════════════════════════════
FEATURE_COLS = [
    'age', 'bmi', 'blood_pressure', 'glucose',
    'cholesterol', 'smoking', 'physical_activity',
    'family_history', 'hba1c', 'creatinine'
]

TARGET_MAP = {
    'diabetes': 'diabetes_risk',
    'heart':    'heart_risk',
    'kidney':   'kidney_risk',
}


def train_and_save(df):
    os.makedirs('models', exist_ok=True)
    results = {}

    for disease, target_col in TARGET_MAP.items():
        print(f"\n── Training {disease.upper()} model ──────────────────")

        X = df[FEATURE_COLS]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        # Random Forest — robust, interpretable via feature importances
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced'
        )
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        acc = accuracy_score(y_test, y_pred)
        print(f"  Accuracy: {acc:.2%}")
        print(classification_report(y_test, y_pred,
                                    target_names=['Low', 'Medium', 'High']))

        # Persist model + scaler together
        bundle = {'model': clf, 'scaler': scaler, 'features': FEATURE_COLS}
        path = f"models/{disease}_model.pkl"
        with open(path, 'wb') as f:
            pickle.dump(bundle, f)
        print(f"  Saved → {path}")

        results[disease] = acc

    return results


# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("🔬 Generating synthetic patient dataset …")
    df = generate_dataset()
    df.to_csv('models/patient_data.csv', index=False)
    print(f"   Dataset: {len(df)} rows × {len(df.columns)} columns")
    print(f"   Saved → models/patient_data.csv")

    print("\n🤖 Training Random Forest classifiers …")
    results = train_and_save(df)

    print("\n✅ All models trained successfully!")
    print("   Accuracies:", {k: f"{v:.1%}" for k, v in results.items()})
    print("\nNext step → python app.py")
