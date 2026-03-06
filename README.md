# 🏥 MediPredict AI — Early Health Risk Prediction System

An AI-powered healthcare web application that predicts early risk of **Diabetes**, **Heart Disease**, and **Kidney Disease** using machine learning.

---

## 🚀 Quick Start (3 commands)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ML models (run once)
python train_model.py

# 3. Launch the web server
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## 📁 Project Structure

```
healthcare-ai/
├── app.py              ← Flask web server + API routes
├── train_model.py      ← Dataset generation + model training
├── requirements.txt    ← Python dependencies
├── models/             ← Trained model .pkl files (generated)
│   ├── diabetes_model.pkl
│   ├── heart_model.pkl
│   └── kidney_model.pkl
├── templates/          ← Jinja2 HTML templates
│   ├── index.html      ← Home page
│   ├── form.html       ← Patient input form
│   └── result.html     ← Prediction results & report
└── static/
    ├── style.css       ← Full CSS (medical-tech theme)
    └── script.js       ← Form validation, animations, demo data
```

---

## 🤖 Machine Learning

| Disease       | Model         | Accuracy | Key Features                         |
|---------------|---------------|----------|--------------------------------------|
| Diabetes      | Random Forest | ~85%     | Glucose, HbA1c, BMI, Family History  |
| Heart Disease | Random Forest | ~85%     | Blood Pressure, Cholesterol, Smoking |
| Kidney Disease| Random Forest | ~86%     | Creatinine, Blood Pressure, Glucose  |

### Why Random Forest?
- Works well on tabular health data
- Provides **feature importances** for Explainable AI
- Robust to outliers and non-linear patterns
- Handles class imbalance via `class_weight='balanced'`

---

## 🌐 API Endpoints

| Method | Route          | Description                                      |
|--------|----------------|--------------------------------------------------|
| GET    | `/`            | Home page                                        |
| GET    | `/form`        | Patient data input form                          |
| GET    | `/results`     | View latest prediction report                    |
| POST   | `/submit_data` | Submit form → run prediction → redirect to results |
| POST   | `/predict`     | JSON prediction API (for testing/integration)    |

### Example API Call

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "name": "John Doe", "gender": "Male",
    "age": 55, "bmi": 29.5, "blood_pressure": 145,
    "glucose": 180, "cholesterol": 240, "hba1c": 7.2,
    "creatinine": 1.8, "smoking": 1,
    "physical_activity": 2, "family_history": 1
  }'
```

### Example Response

```json
{
  "success": true,
  "report": {
    "overall_risk": "High",
    "doctor_alert": true,
    "predictions": {
      "diabetes": {
        "risk_level": "High",
        "probabilities": { "Low": 2.5, "Medium": 18.0, "High": 79.5 },
        "top_factors": [
          { "feature": "Glucose Level", "importance": 38.4 },
          { "feature": "HbA1c (Diabetes marker)", "importance": 31.2 }
        ],
        "recommendations": ["Consult an endocrinologist immediately.", "..."]
      }
    }
  }
}
```

---

## 🎯 Features

- **3-disease screening** — Diabetes, Heart Disease, Kidney Disease
- **Risk levels** — Low / Medium / High
- **Doctor alert** — Red warning banner if any disease is High risk
- **Explainable AI** — Visual bar charts of top influencing health factors
- **Probability bars** — Shows % confidence for each risk level
- **Personalised recommendations** — 3–5 actionable steps per disease
- **Demo mode** — One-click "fill high-risk sample" button for hackathon demos
- **Print-ready** — Print or save as PDF from the results page
- **Fully responsive** — Works on mobile, tablet, desktop

---

## 🩺 Input Fields

| Field             | Unit     | Normal Range    |
|-------------------|----------|-----------------|
| Age               | years    | 18–100          |
| BMI               | kg/m²    | 18.5–24.9       |
| Blood Pressure    | mmHg     | < 120 systolic  |
| Fasting Glucose   | mg/dL    | 70–99           |
| Total Cholesterol | mg/dL    | < 200           |
| HbA1c             | %        | < 5.7           |
| Serum Creatinine  | mg/dL    | 0.6–1.2         |
| Physical Activity | days/wk  | ≥ 5 recommended |
| Smoking           | 0/1      | 0 = Non-smoker  |
| Family History    | 0/1      | 0 = No history  |

---

## ⚠️ Medical Disclaimer

This system is a **screening tool for educational and hackathon purposes only**. It does not constitute medical advice, diagnosis, or treatment. The synthetic dataset used for training does not represent real patient data. Always consult a qualified healthcare professional for any medical decisions.

---

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, Vanilla JS (no framework needed)
- **Backend**: Python 3.10+, Flask 3.x
- **ML**: Scikit-learn (Random Forest Classifier)
- **Data**: Pandas, NumPy (synthetic dataset generation)
- **Fonts**: Google Fonts — DM Serif Display + DM Sans
