import pickle
import os
from src.preprocess import clean_text

def load_model(path: str = 'models/sentiment_model.pkl'):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}\nfirst train the model: python src/train_model.py")
    with open(path, 'rb') as f:
        return pickle.load(f)


def predict_single(text: str, model=None) -> dict:
    """ Single tweet predict"""
    if model is None:
        model = load_model()
    cleaned = clean_text(text)
    sentiment = model.predict([cleaned])[0]

    # Probability 
    try:
        probs = model.predict_proba([cleaned])[0]
        classes = model.classes_
        confidence = {c: round(float(p) * 100, 1) for c, p in zip(classes, probs)}
    except AttributeError:
        confidence = {sentiment: 100.0}

    return {
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence
    }


def predict_batch(texts: list, model=None) -> list:
    """Multiple tweets predict"""
    if model is None:
        model = load_model()
    cleaned = [clean_text(t) for t in texts]
    predictions = model.predict(cleaned)
    return [{'text': t, 'sentiment': s} for t, s in zip(texts, predictions)]


if __name__ == '__main__':
    model = load_model()
    test_tweets = [
        "I absolutely love this game! Best purchase ever!",
        "This product is terrible, complete waste of money",
        "Just downloaded the new update, seems okay",
    ]
    print("\n--- Prediction Results ---")
    for tweet in test_tweets:
        result = predict_single(tweet, model)
        print(f"\nText: {result['text'][:60]}...")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']}")
