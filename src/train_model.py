"""
src/train_model.py
------------------
Model train karo aur save karo
Run: python src/train_model.py
"""

import pickle
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from src.preprocess import load_and_clean


def build_pipeline(model_name: str = 'logistic'):
    models = {
        'logistic':  LogisticRegression(max_iter=500, C=1.0),
        'naive_bayes': MultinomialNB(alpha=0.1),
        'svm':       LinearSVC(max_iter=1000)
    }
    return Pipeline([
        ('tfidf', TfidfVectorizer(max_features=15000, ngram_range=(1, 2))),
        ('clf',   models[model_name])
    ])


def train_and_evaluate():
    print("=" * 50)
    print("  Sentiment Analysis — Model Training")
    print("=" * 50)

    # Data load karo
    print("\n[1/4] Loading data...")
    train = load_and_clean('data/twitter_training.csv')
    val   = load_and_clean('data/twitter_validation.csv')
    print(f"  Train: {len(train):,} | Val: {len(val):,}")

    # Models compare karo
    results = {}
    print("\n[2/4] Training models...")
    for name in ['logistic', 'naive_bayes', 'svm']:
        pipe = build_pipeline(name)
        pipe.fit(train['clean_text'], train['sentiment'])
        preds = pipe.predict(val['clean_text'])
        acc = accuracy_score(val['sentiment'], preds)
        results[name] = {'accuracy': round(acc * 100, 2), 'pipeline': pipe}
        print(f"  {name:<15} Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Best model select karo
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best_pipe  = results[best_name]['pipeline']
    print(f"\n[3/4] Best model: {best_name} ({results[best_name]['accuracy']}%)")

    # Detailed report
    preds = best_pipe.predict(val['clean_text'])
    print("\n--- Classification Report ---")
    print(classification_report(val['sentiment'], preds))

    # Save karo
    print("[4/4] Saving model...")
    os.makedirs('models', exist_ok=True)
    with open('models/sentiment_model.pkl', 'wb') as f:
        pickle.dump(best_pipe, f)

    # Metrics JSON mein save karo
    metrics = {k: v['accuracy'] for k, v in results.items()}
    metrics['best_model'] = best_name
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Model saved → models/sentiment_model.pkl")
    print(f"  Metrics saved → models/metrics.json")
    print("\nDone! ✓")
    return best_pipe


if __name__ == '__main__':
    train_and_evaluate()
