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
