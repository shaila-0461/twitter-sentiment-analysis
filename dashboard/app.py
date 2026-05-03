"""
dashboard/app.py
----------------
Flask web application — Twitter Sentiment Dashboard
Run: python dashboard/app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import json
import pickle
import re
from collections import Counter
from flask import Flask, render_template, request, jsonify
import pandas as pd

app = Flask(__name__)

# ── Globals ──────────────────────────────────────────────────────────────────
MODEL = None
TRAIN_DF = None


def get_model():
    global MODEL
    if MODEL is None:
        try:
            with open(os.path.join(os.path.dirname(__file__), '../models/sentiment_model.pkl'), 'rb') as f:
                MODEL = pickle.load(f)
        except FileNotFoundError:
            MODEL = None
    return MODEL


def get_data():
    global TRAIN_DF
    if TRAIN_DF is None:
        path = os.path.join(os.path.dirname(__file__), '../data/twitter_training.csv')
        df = pd.read_csv(path, header=None)
        df.columns = ['id', 'topic', 'sentiment', 'text']
        TRAIN_DF = df[df['sentiment'] != 'Irrelevant'].copy()
    return TRAIN_DF


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+|#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/stats')
def stats():
    df = get_data()
    total = len(df)
    overall = df['sentiment'].value_counts().to_dict()
    topics = df['topic'].value_counts().head(6).index.tolist()

    sentiment_by_topic = {}
    for t in topics:
        sub = df[df['topic'] == t]['sentiment'].value_counts().to_dict()
        sentiment_by_topic[t] = {
            'Positive': sub.get('Positive', 0),
            'Negative': sub.get('Negative', 0),
            'Neutral':  sub.get('Neutral',  0)
        }

    return jsonify({
        'total_tweets':      total,
        'overall_sentiment': overall,
        'sentiment_by_topic': sentiment_by_topic,
        'top_topics':        topics,
        'all_topics':        df['topic'].unique().tolist(),
        'model_accuracy':    91.18
    })

@app.route('/api/live_tweets', methods=['POST'])
def live_tweets():
    from twitter_api import analyze_live_tweets
    data  = request.get_json()
    query = data.get('query', 'technology')
    count = int(data.get('count', 10))

    df = analyze_live_tweets(query, count)
    if df.empty:
        return jsonify({'error': 'no tweets found'}), 404

    return jsonify(df.to_dict(orient='records'))


@app.route('/api/wordcloud')
def wordcloud():
    df = get_data()
    sentiment = request.args.get('sentiment', 'Positive')
    topic     = request.args.get('topic', 'all')

    sub = df if topic == 'all' else df[df['topic'] == topic]
    sub = sub[sub['sentiment'] == sentiment]

    stopwords = {'the','a','an','is','are','was','i','to','of','and','in',
                 'it','for','on','this','that','with','at','by','or','but',
                 'not','so','if','we','you','he','she','they','just','get',
                 'im','dont','have','been','will','can','be','my','its'}

    words = []
    for text in sub['text'].str.lower():
        ws = re.findall(r'\b[a-z]{3,}\b', str(text))
        words.extend([w for w in ws if w not in stopwords])

    top = Counter(words).most_common(40)
    return jsonify([{'word': w, 'count': c} for w, c in top])


@app.route('/api/predict', methods=['POST'])
def predict():
    data  = request.get_json()
    text  = data.get('text', '')
    model = get_model()

    if model is None:
        return jsonify({'error': 'Model not loaded. Run train_model.py first.'}), 500

    cleaned   = clean_text(text)
    sentiment = model.predict([cleaned])[0]

    try:
        probs      = model.predict_proba([cleaned])[0]
        classes    = model.classes_
        confidence = {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)}
    except Exception:
        confidence = {sentiment: 100.0}

    return jsonify({'sentiment': sentiment, 'confidence': confidence, 'text': text})


@app.route('/api/topic_trend')
def topic_trend():
    df    = get_data()
    topic = request.args.get('topic', 'all')
    sub   = df if topic == 'all' else df[df['topic'] == topic]
    dist  = sub['sentiment'].value_counts().to_dict()
    return jsonify(dist)


if __name__ == '__main__':
    print("\n Twitter Sentiment Analysis \n")
    print(" Open: http://localhost:5000\n")
    app.run(debug=True, port=5000)
