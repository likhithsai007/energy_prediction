# Smart Energy Consumption Forecasting ⚡

A clean, full-stack Machine Learning web application designed to forecast household electricity consumption, identify peak demand periods, analyze energy patterns with K-Means clustering, provide smart load-shifting recommendations, and compare regression model performance.

---

## 🏗️ Architecture & Technology Stack

- **Frontend**: React 18, Vite, Recharts, Lucide Icons, Modern Vanilla CSS Design System.
- **Backend**: Python 3.11+, FastAPI, Uvicorn, scikit-learn, pandas, NumPy, joblib.
- **Dataset**: UCI Appliances Energy Prediction Dataset (`energydata_complete.csv`).

---

## 📁 Project Structure

```text
.
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Forecast.jsx
│   │   │   └── ModelAnalysis.jsx
│   │   ├── components/
│   │   │   ├── ForecastCard.jsx
│   │   │   ├── PeakCard.jsx
│   │   │   ├── ConsumptionChart.jsx
│   │   │   └── ModelTable.jsx
│   │   ├── services/
│   │   │   └── api.js
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   └── index.html
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── routes/
│   │   │   ├── forecast.py
│   │   │   └── models.py
│   │   ├── services/
│   │   │   ├── forecasting.py
│   │   │   ├── model_selection.py
│   │   │   └── recommendations.py
│   │   └── models/
│   │       ├── train.py
│   │       └── trained_models/
│   │           ├── linear_regression.joblib
│   │           ├── decision_tree.joblib
│   │           ├── random_forest.joblib
│   │           ├── kmeans.joblib
│   │           ├── scaler.joblib
│   │           └── model_metadata.json
│   ├── data/
│   │   └── energydata_complete.csv
│   └── requirements.txt
│
├── energydata_complete.csv
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Backend Setup & Startup

From the project root:

```bash
# Install Python dependencies
pip install -r backend/requirements.txt

# (Optional) Train models if not already pre-trained
python backend/app/models/train.py

# Start FastAPI server (runs at http://localhost:8000)
cd backend
uvicorn app.main:app --reload --port 8000
```

### 2. Frontend Setup & Startup

In a separate terminal window:

```bash
cd frontend

# Install npm dependencies
npm install

# Launch Vite development server (runs at http://localhost:5173)
npm run dev
```

Visit **`http://localhost:5173`** in your browser.

---

## 🤖 Machine Learning Workflow

1. **Supervised Regression**:
   - **Linear Regression**: Baseline linear benchmark.
   - **Decision Tree Regressor**: Tree-based non-linear model.
   - **Random Forest Regressor**: Ensemble learning model achieving the highest R² score and lowest MAE/RMSE.
   - *Model Selection*: The backend automatically identifies and deploys the highest-performing model.

2. **Unsupervised Pattern Analysis**:
   - **K-Means Clustering ($k=3$)**: Segmenting usage profiles into **Low**, **Medium**, and **High Consumption** patterns evaluated via Silhouette Score.

3. **Smart Load Recommendation**:
   - Recommends actionable load-shifting actions only when high consumption or peak intensity demands it.

---

## 🔌 API Reference

- `GET /api/health`: Health status check.
- `POST /api/forecast`: Generates 10-minute continuous predictions, peak periods, total kWh, and recommendations.
- `GET /api/models/performance`: Evaluation metrics (MAE, RMSE, R², MAPE) across models.
- `GET /api/patterns`: K-Means cluster profiles and Silhouette Score.
- `GET /api/models/feature-importance`: Top predictive features for the chosen model.
