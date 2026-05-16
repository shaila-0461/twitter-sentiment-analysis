import pandas as pd
import joblib
import sys
import os

# PATH FIX 
print("Current Directory:", os.getcwd())

# add src in path for imports
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Import from src
from preprocess import clean_text

LABEL_REVERSE = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

#  TWITTER CLIENT 
def get_client():
    import tweepy
    API_KEY = "rc2tl1p58ky0qATRGMahjP8IL"
    API_SECRET = "tBouShWf4K07jXJlVFVBZdyewGCrt0Wz3nH6IKuG7YQJLIUUU9"
    ACCESS_TOKEN = "2050809271617826816-j7uBRHXA2QyWxe1vVOhRY4FSweuNq0"
    ACCESS_TOKEN_SECRET = "wdytLXouh09aRiFmrQcGFVZEu2wW8n2nEiKvJpU3RDWAf"
    BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAHCN9QEAAAAAovk5ueqoOlmcah0mGYHUeZ6mKAE%3DhgLRKBnr711H2KIVOTgxh3oEmpVNC4BaVuEFYaWkXG7A6kXeVi"

    return tweepy.Client(
        bearer_token=BEARER_TOKEN,
        consumer_key=API_KEY,
        consumer_secret=API_SECRET,
        access_token=ACCESS_TOKEN,
        access_token_secret=ACCESS_TOKEN_SECRET
    )


# MAIN FUNCTION 
def analyze_live_tweets(query: str, count: int = 20) -> pd.DataFrame:
    try:
        print(f"Fetching tweets for: {query}")
        client = get_client()
        
        response = client.search_recent_tweets(
            query=f"{query} -is:retweet lang:en",
            max_results=min(count, 100),
            tweet_fields=['text', 'created_at']
        )

        if not response.data:
            print("No tweets found")
            return pd.DataFrame()

        df = pd.DataFrame([{'text': t.text, 'created_at': t.created_at} for t in response.data])

        # Model Load
        model_paths = ['models/best_model.pkl', 'models/sentiment_model.pkl', 'sentiment_model.pkl']
        model = None
        for path in model_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                print(f" Model loaded from: {path}")
                break

        if model is None:
            print(" Model not found!")
            return pd.DataFrame()

        df['clean_text'] = df['text'].apply(clean_text)
        df['sentiment'] = model.predict(df['clean_text'])
        df['sentiment'] = df['sentiment'].map(LABEL_REVERSE)

        print(f" Success: {len(df)} tweets analyzed")
        return df[['text', 'sentiment', 'created_at']]

    except Exception as e:
        print(f" ERROR: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()
