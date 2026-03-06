"""
app.py — Flask Backend for AI Health Risk Prediction System
Serves HTML pages and exposes prediction API endpoints.
"""

from flask import Flask, render_template, request, jsonify, session
import pickle
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'healthcare-ai-secret-2024'   # session encryption key

# ─── Load ML models at startup ───────────────────────────────────────────────
MODELS = {}
MODEL_FILES = {
    'diabetes': 'models/diabetes_model.pkl',
    'heart':    'models/heart_model.pkl',
    'kidney':   'models/kidney_model.pkl',
}

for disease, path in MODEL_FILES.items():
    if os.path.exists(path):
        with open(path, 'rb') as f:
            MODELS[disease] = pickle.load(f)
        print(f"✅ Loaded {disease} model")
    else:
        print(f"⚠️  Model not found: {path}  →  run train_model.py first")

FEATURE_COLS = [
    'age', 'bmi', 'blood_pressure', 'glucose',
    'cholesterol', 'smoking', 'physical_activity',
    'family_history', 'hba1c', 'creatinine'
]

RISK_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

# Feature display names for explainability panel
FEATURE_DISPLAY = {
    'age':               'Age',
    'bmi':               'BMI',
    'blood_pressure':    'Blood Pressure',
    'glucose':           'Glucose Level',
    'cholesterol':       'Cholesterol',
    'smoking':           'Smoking',
    'physical_activity': 'Physical Activity',
    'family_history':    'Family History',
    'hba1c':             'HbA1c (Diabetes marker)',
    'creatinine':        'Creatinine (Kidney marker)',
}

# ─── Recommendation library ───────────────────────────────────────────────────
RECOMMENDATIONS = {
    'diabetes': {
        'Low': [
            "Maintain a balanced diet rich in fibre and low in refined sugars.",
            "Continue regular aerobic exercise (150 min/week recommended).",
            "Monitor blood glucose annually as a preventive measure.",
        ],
        'Medium': [
            "Reduce daily sugar and processed-carbohydrate intake significantly.",
            "Aim for at least 30 minutes of moderate exercise 5 days/week.",
            "Schedule HbA1c and fasting glucose tests every 6 months.",
            "Consider a consultation with a dietitian for a personalised meal plan.",
        ],
        'High': [
            "⚠️ Consult an endocrinologist or GP immediately.",
            "Begin a strict low-glycaemic diet — avoid sugary drinks entirely.",
            "Self-monitor blood glucose daily if prescribed by your doctor.",
            "Discuss Metformin or other preventive medication options with your doctor.",
            "Enrol in a structured diabetes prevention programme.",
        ],
    },
    'heart': {
        'Low': [
            "Keep up heart-healthy habits: oily fish, nuts, olive oil.",
            "Stay physically active and manage stress effectively.",
            "Check blood pressure and cholesterol annually.",
        ],
        'Medium': [
            "Reduce saturated fat and sodium intake.",
            "If you smoke, start a cessation programme today.",
            "Aim for BP below 130/80 mmHg — monitor at home weekly.",
            "Consider a cardiology screening (ECG, lipid panel).",
        ],
        'High': [
            "⚠️ Seek immediate cardiology evaluation.",
            "Stop smoking — it is the single most impactful step you can take.",
            "Begin a medically supervised cardiac rehabilitation programme.",
            "Discuss statin therapy and antihypertensives with your cardiologist.",
            "Follow a strict DASH or Mediterranean diet.",
        ],
    },
    'kidney': {
        'Low': [
            "Stay well hydrated (2–3 litres of water daily).",
            "Limit NSAIDs (ibuprofen, etc.) to protect kidney function.",
            "Annual creatinine and eGFR blood tests are advisable.",
        ],
        'Medium': [
            "Reduce dietary protein and sodium intake.",
            "Strictly control blood pressure and blood sugar.",
            "Test kidney function (eGFR, urine albumin) every 3–6 months.",
            "Avoid nephrotoxic medications without specialist advice.",
        ],
        'High': [
            "⚠️ Consult a nephrologist without delay.",
            "Follow a kidney-protective diet (low potassium, low phosphorus).",
            "Monitor fluid intake carefully and avoid dehydration.",
            "Discuss dialysis planning if eGFR is critically low.",
            "Review all medications for kidney safety with your doctor.",
        ],
    },
}


# ════════════════════════════════════════════════════════════════════════════
# ROUTES — PAGES
# ════════════════════════════════════════════════════════════════════════════

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/results')
def results():
    """Read prediction stored in session and render results page."""
    prediction = session.get('prediction')
    if not prediction:
        return render_template('result.html', error=True)
    return render_template('result.html', prediction=prediction)


# ════════════════════════════════════════════════════════════════════════════
# ROUTES — API
# ════════════════════════════════════════════════════════════════════════════

@app.route('/submit_data', methods=['POST'])
def submit_data():
    """
    Receive JSON patient data from the form, run predictions,
    store in session, return JSON for client-side redirect.
    """
    try:
        data = request.get_json()
        features = extract_features(data)
        predictions = run_predictions(features)

        # Build full report
        report = build_report(data, features, predictions)

        # Persist in session so /results page can read it
        session['prediction'] = report

        return jsonify({'success': True, 'redirect': '/results'})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Direct JSON prediction endpoint (for API / testing).
    Returns full prediction JSON without page redirect.
    """
    try:
        data = request.get_json()
        features = extract_features(data)
        predictions = run_predictions(features)
        report = build_report(data, features, predictions)
        return jsonify({'success': True, 'report': report})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ════════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════════

def extract_features(data: dict):
    """Parse and validate patient data, return pandas DataFrame (1 row)."""
    import pandas as pd
    row = {}
    for col in FEATURE_COLS:
        val = data.get(col)
        if val is None:
            raise ValueError(f"Missing field: {col}")
        row[col] = float(val)
    return pd.DataFrame([row])


def run_predictions(features: np.ndarray) -> dict:
    """Run all disease models and return structured results."""
    results = {}

    for disease, bundle in MODELS.items():
        model  = bundle['model']
        scaler = bundle['scaler']

        X_scaled = scaler.transform(features)
        pred_class = int(model.predict(X_scaled)[0])
        pred_proba = model.predict_proba(X_scaled)[0].tolist()

        # Feature importances from Random Forest
        importances = model.feature_importances_.tolist()
        feat_imp = sorted(
            zip(FEATURE_COLS, importances),
            key=lambda x: x[1], reverse=True
        )

        results[disease] = {
            'risk_level':       RISK_LABELS[pred_class],
            'risk_score':       pred_class,
            'probabilities': {
                'Low':    round(pred_proba[0] * 100, 1),
                'Medium': round(pred_proba[1] * 100, 1),
                'High':   round(pred_proba[2] * 100, 1),
            },
            'top_factors': [
                {'feature': FEATURE_DISPLAY[f], 'importance': round(imp * 100, 1)}
                for f, imp in feat_imp[:5]
            ],
            'recommendations': RECOMMENDATIONS[disease][RISK_LABELS[pred_class]],
        }

    return results


def build_report(raw_data: dict, features: np.ndarray, predictions: dict) -> dict:
    """Assemble the full patient report dict."""
    # Overall risk = max across diseases
    max_score = max(v['risk_score'] for v in predictions.values())
    overall_risk = RISK_LABELS[max_score]

    return {
        'patient': {
            'name':  raw_data.get('name', 'Patient'),
            'age':   int(raw_data.get('age', 0)),
            'gender': raw_data.get('gender', '—'),
        },
        'timestamp':    datetime.now().strftime('%d %B %Y, %H:%M'),
        'overall_risk': overall_risk,
        'predictions':  predictions,
        'doctor_alert': overall_risk == 'High',
        'vitals': {
            'BMI':             raw_data.get('bmi'),
            'Blood Pressure':  f"{raw_data.get('blood_pressure')} mmHg",
            'Glucose':         f"{raw_data.get('glucose')} mg/dL",
            'Cholesterol':     f"{raw_data.get('cholesterol')} mg/dL",
            'HbA1c':           raw_data.get('hba1c'),
            'Creatinine':      raw_data.get('creatinine'),
        }
    }


# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    if not MODELS:
        print("\n❌  No models loaded — please run:  python train_model.py\n")
    else:
        print(f"\n🏥  Healthcare AI running at http://127.0.0.1:5000\n")
        app.run(debug=True, port=5000)
