# 🗳️ Tamil Nadu Assembly Election 2026 — Data Analysis & Winner Prediction

![Python](https://img.shields.io/badge/Python-3.14-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange?logo=scikit-learn)
![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-red)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

A complete end-to-end Data Science project analyzing the **Tamil Nadu Assembly
Election Results 2026** from the Election Commission of India.

This project covers the full data science workflow:
- Data Cleaning & Feature Engineering
- Exploratory Data Analysis (EDA)
- 12 Professional Visualizations
- Machine Learning — Winner Prediction
- Model Evaluation & Comparison

**Best Model: XGBoost — ROC-AUC: 0.9961 | Accuracy: 98.24%**

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | Election Commission of India (ECI) |
| Election | Tamil Nadu Assembly Election 2026 |
| Total Records | 4,257 candidate-level results |
| Constituencies | 234 |
| Candidates | 3,872 |
| Parties | 105 |
| Missing Values | 0 (100% clean) |

---

## 🔍 Key Insights

- **TVK won 107/234 seats** — massive debut for a new party
- **DMK: 60 seats | AIADMK: 47 seats** — traditional parties far behind
- **Closest contest**: TIRUPPATTUR — won by just **30 votes**
- **Biggest win**: EDAPPADI — won by **98,110 votes**
- **Average winning margin**: 16,784 votes
- **94.5% of candidates lost** — highly imbalanced, handled with class weighting

---

## 🤖 ML Results

| Model | ROC-AUC | Accuracy |
|---|---|---|
| 🥇 XGBoost | **0.9961** | **98.24%** |
| Gradient Boosting | 0.9953 | 97.89% |
| Logistic Regression | 0.9951 | 97.42% |
| Random Forest | 0.9942 | 97.77% |

**Top Features** (by importance):
1. `pct_votes` — vote share percentage
2. `total_votes` — raw vote count
3. `party_win_rate` — historical party strength
4. `is_major_party` — major party advantage
5. `is_independent` — independents rarely win

---

## 📁 Project Structure

---

## 🚀 How to Run

```bash
# 1. Clone the repo
git clone https://github.com/Eswarpadala-lgtm/tamil-nadu-election-2026-analysis.git
cd tamil-nadu-election-2026-analysis

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows

# 3. Install libraries
pip install -r requirements.txt

# 4. Run the pipeline
python src/data_cleaning.py
python src/eda_visualization.py
python src/ml_model.py
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.14 | Core language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib + Seaborn | Visualizations |
| Scikit-learn | ML models + evaluation |
| XGBoost | Best performing model |
| Joblib | Model persistence |
| Git + GitHub | Version control |

---

## 📈 Visualizations (12 Charts)

| # | Chart | Insight |
|---|---|---|
| 01 | Seats Won by Party | TVK dominance |
| 02 | Vote Share Distribution | Party-wise spread |
| 03 | Winning Margin Distribution | Election competitiveness |
| 04 | Biggest Wins & Closest Contests | Dramatic results |
| 05 | Winner vs Loser Vote Share | Class separation |
| 06 | Candidates per Constituency | Competition level |
| 07 | Correlation Heatmap | Feature relationships |
| 08 | Party Win Rate | Efficiency analysis |
| 09 | Model Comparison | XGBoost wins |
| 10 | Confusion Matrix | Prediction accuracy |
| 11 | ROC Curves | All models compared |
| 12 | Feature Importance | What drives winning |

---

## 👨‍💻 Author

**Eswar** — Data Science Enthusiast
- GitHub: [@Eswarpadala-lgtm](https://github.com/Eswarpadala-lgtm)

---

## 📄 License

MIT License — free to use and learn from.