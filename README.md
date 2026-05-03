# Twitter Sentiment Analysis Dashboard

**AI Course — Semester Project**

## Project Overview

A web-based **Sentiment Analysis Dashboard** that classifies tweets as **Positive**, **Negative**, or **Neutral** using Machine Learning.

- **Dataset**: 74,682 training tweets + 1,758 validation tweets
- **Model**: TF-IDF + Machine Learning Classifiers
- **Best Accuracy**: 91.18%
- **Tech Stack**: Python, Flask, scikit-learn, Chart.js, Tweepy

---

## Features

- Real-time sentiment prediction for any tweet
- Interactive dashboard with charts and visualizations
- Word cloud analysis by sentiment
- Live Twitter data fetching and analysis (using X API)
- Model performance comparison (Logistic Regression, SVM, Naive Bayes)

---

## Folder Structure
twitter-sentiment-analysis/
│
├── data/                          # Datasets
│   ├── twitter_training.csv
│   └── twitter_validation.csv
│
├── src/                           # Core Python code
│   ├── preprocess.py              # Text cleaning
│   ├── train_model.py             # Model training
│   └── predict.py                 # Prediction utilities
│
├── models/                        # Trained models
│   ├── sentiment_model.pkl
│   └── metrics.json
│
├── dashboard/                     # Web Application
│   ├── app.py                     # Flask backend
│   └── templates/
│       └── index.html             # Frontend (Dashboard)
│
├── notebooks/                     # Jupyter Notebooks (EDA)
├── requirements.txt
└── README.md


---

## File Purpose

| File                        | Description |
|---------------------------|-----------|
| `src/preprocess.py`       | Cleans raw tweet text (removes URLs, mentions, stopwords, etc.) |
| `src/train_model.py`      | Trains multiple models and saves the best one |
| `src/predict.py`          | Utility for single and batch predictions |
| `dashboard/app.py`        | Main Flask web server with all API endpoints |
| `dashboard/templates/index.html` | Interactive frontend dashboard |
| `data/twitter_training.csv` | Main training dataset |

---

## Setup & Installation

### 1. Clone the Repository
### 2. Install Dependencies
pip install -r requirements.txt

### 3.Train the Model
python src/train_model.py

### 4.Run the Dashboard
python dashboard/app.py

Open your browser and go to: http://localhost:5000

Dashboard Features

Overall sentiment distribution (Doughnut Chart)
Sentiment analysis by topic (Stacked Bar Chart)
Top topics by tweet volume
Dynamic word clouds for each sentiment
Live tweet predictor (type any text)
Real-time Twitter fetching and sentiment analysis
Model performance metrics


Dataset Information

Sentiments: Positive, Negative, Neutral, Irrelevant
Topics: Microsoft, Google, Amazon, Verizon, gaming companies, etc.
Only relevant (non-Irrelevant) tweets are used for training