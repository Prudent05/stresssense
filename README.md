# StressSense — Early Financial Stress Signal System

StressSense is a full-stack financial risk application designed to detect **early financial stress signals** before default occurs.

Unlike traditional credit models that only predict binary default outcomes, StressSense focuses on **early-stage behavioral shifts** (utilization pressure, missed payments, spending instability, income consistency) and translates them into a clear, consumer-friendly risk score and action plan.

The system combines a machine learning backend with a responsive web interface to deliver real-time financial stress insights.

---

## 🎯 Problem Statement

Most credit risk systems focus on predicting whether a customer will default.  
However, by the time default is predicted, it is often too late for meaningful intervention.

StressSense reframes the problem:

> Can we detect early behavioral signals of financial stress and surface them in a way that changes user decisions before risk escalates?

This makes the system more product-relevant and intervention-focused rather than purely predictive.

---

## 🧠 System Overview

### Inputs (User-Provided Signals)

- Credit Utilization (% of limit used)
- Missed Payments (recent count)
- Spending Volatility (low / medium / high)
- Income Stability (low / medium / high)

### Model Output

- `stress_score` (0–100)
- `stress_tier`
  - Stable
  - Watchlist
  - High Stress
- `prob_default` (model probability)
- `top_reasons` (plain-language risk drivers)
- `actions` (recommended intervention steps)

---

## 🏗 Architecture

# StressSense — Early Financial Stress Signal System

StressSense is a full-stack financial risk application designed to detect **early financial stress signals** before default occurs.

Unlike traditional credit models that only predict binary default outcomes, StressSense focuses on **early-stage behavioral shifts** (utilization pressure, missed payments, spending instability, income consistency) and translates them into a clear, consumer-friendly risk score and action plan.

The system combines a machine learning backend with a responsive web interface to deliver real-time financial stress insights.

---

## 🎯 Problem Statement

Most credit risk systems focus on predicting whether a customer will default.  
However, by the time default is predicted, it is often too late for meaningful intervention.

StressSense reframes the problem:

> Can we detect early behavioral signals of financial stress and surface them in a way that changes user decisions before risk escalates?

This makes the system more product-relevant and intervention-focused rather than purely predictive.

---

## 🧠 System Overview

### Inputs (User-Provided Signals)

- Credit Utilization (% of limit used)
- Missed Payments (recent count)
- Spending Volatility (low / medium / high)
- Income Stability (low / medium / high)

### Model Output

- `stress_score` (0–100)
- `stress_tier`
  - Stable
  - Watchlist
  - High Stress
- `prob_default` (model probability)
- `top_reasons` (plain-language risk drivers)
- `actions` (recommended intervention steps)

---

## 🏗 Architecture
Frontend (HTML + Tailwind)
↓
FastAPI Backend
↓
UI Model (scikit-learn Pipeline)
↓
Prediction + Business Logic Layer
↓
Consumer-Friendly Output


---

## 📊 Modeling Approach

- Dataset source: Kaggle – American Express Default Prediction
- Snapshot engineering to avoid data leakage
- Feature transformation into simplified proxy variables for UI consumption
- Logistic-based probability model trained on engineered snapshot
- Tier mapping based on scaled probability output

The model is not exposed directly to users.  
Instead, outputs are translated into actionable behavioral guidance.

---

## 🚀 Features

### 1️⃣ Assessment Page
Users answer 4 core questions and immediately receive:
- Stress score
- Tier classification
- Top reasons
- Recommended actions

### 2️⃣ Dashboard
Displays:
- Current stress score
- Risk probability
- Behavioral drivers
- Action plan

Inputs are stored locally and re-evaluated dynamically.

### 3️⃣ What-If Simulator
Allows users to simulate:
- Lowering utilization
- Reducing missed payments
- Stabilizing spending
- Increasing income consistency

Results update in near real-time via API calls.

---

## ⚙️ Tech Stack

**Backend**
- Python
- FastAPI
- Uvicorn
- scikit-learn
- Pandas / NumPy

**Frontend**
- HTML
- TailwindCSS
- Vanilla JavaScript

**Modeling**
- Feature engineering pipeline
- Snapshot-based training approach
- Probability-to-tier transformation logic

---

## ▶️ Running Locally

```bash
python -m uvicorn app.main:app --reload

