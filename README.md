# Twitter Sentiment Analysis Project
**AI Course — Semester Project**

## Project Overview
Social media tweets ko Positive / Negative / Neutral classify karna using Machine Learning.

- **Dataset:** Twitter Training (74,682 tweets) + Validation (1,758 tweets)
- **Model:** Logistic Regression + TF-IDF
- **Accuracy:** 91.18% on validation set
- **Dashboard:** Flask + Chart.js

---

## Folder Structure

```
sentiment_project/
│
├── data/                        ← Datasets
│   ├── twitter_training.csv     ← 74,682 labeled tweets (training)
│   └── twitter_validation.csv  ← 1,758 labeled tweets (testing)
│
├── src/                         ← Python source code
│   ├── preprocess.py            ← Text cleaning pipeline
│   ├── train_model.py           ← Model training + evaluation
│   └── predict.py               ← Single/batch prediction
│
├── models/                      ← Saved model files
│   ├── sentiment_model.pkl      ← Trained model (generate karna hoga)
│   └── metrics.json             ← Accuracy comparison
│
├── dashboard/                   ← Web dashboard
│   ├── app.py                   ← Flask server (main app)
│   └── templates/
│       └── index.html           ← Dashboard UI
│
├── notebooks/                   ← Jupyter notebooks (EDA etc)
├── requirements.txt             ← Python dependencies
└── README.md                    ← Ye file
```

---

## Konsa File Kya Karta Hai?

| File | Kaam |
|------|------|
| `src/preprocess.py` | Raw tweet text clean karta hai (links, symbols, stopwords remove) |
| `src/train_model.py` | 3 models train karta hai, best save karta hai |
| `src/predict.py` | Naye tweet ka sentiment predict karta hai |
| `dashboard/app.py` | Flask web server — API endpoints provide karta hai |
| `dashboard/templates/index.html` | Dashboard UI — charts, wordcloud, live predictor |
| `data/twitter_training.csv` | Training dataset |
| `data/twitter_validation.csv` | Validation/test dataset |

---

## Setup aur Run Karna

### Step 1: Dependencies Install Karo
```bash
pip install -r requirements.txt
```

### Step 2: Model Train Karo
```bash
python src/train_model.py
```
Output:
```
logistic       Accuracy: 0.9118 (91.18%)
naive_bayes    Accuracy: 0.8540 (85.40%)
svm            Accuracy: 0.8970 (89.70%)
Best model: logistic
Model saved → models/sentiment_model.pkl
```

### Step 3: Dashboard Chalao
```bash
python dashboard/app.py
```
Browser mein kholo: **http://localhost:5000**

---

## Dashboard Features
- **Stats Cards:** Total tweets, Positive/Negative/Neutral count
- **Doughnut Chart:** Overall sentiment distribution
- **Stacked Bar Chart:** Sentiment by topic
- **Topic Count Chart:** Top 8 topics
- **Word Cloud:** Most common words per sentiment
- **Model Metrics:** 3 models accuracy comparison
- **Live Predictor:** Apna tweet type karo, real-time prediction pao

---

## Dataset Info
- **Sentiments:** Positive, Negative, Neutral, Irrelevant
- **Topics:** Microsoft, Verizon, Google, Amazon, Gaming companies, etc.
- **Columns:** `id, topic, sentiment, text`
- Irrelevant tweets training se remove kar diye

---

*Submitted by: AI Course Students | April 2026*
