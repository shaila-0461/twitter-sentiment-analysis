"""
src/preprocess.py
-----------------
Text cleaning aur preprocessing pipeline
"""

import re
import pandas as pd


STOPWORDS = {
    'the','a','an','is','are','was','were','i','my','to','of','and','in',
    'it','for','on','this','that','be','have','do','with','at','by','from',
    'or','but','not','so','if','as','we','you','he','she','they','me','him',
    'her','us','our','its','can','will','just','get','got','im','ive','id',
    'dont','cant','wont','didnt','doesnt','isnt'
}


def clean_text(text: str) -> str:
    """Single tweet ko clean karo"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)          # URLs hatao
    text = re.sub(r'@\w+', '', text)                     # Mentions hatao
    text = re.sub(r'#(\w+)', r'\1', text)                # Hashtag symbol hatao
    text = re.sub(r'[^a-z\s]', '', text)                 # Special chars hatao
    text = re.sub(r'\s+', ' ', text).strip()             # Extra spaces hatao
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return ' '.join(words)


def load_and_clean(filepath: str, has_header: bool = False) -> pd.DataFrame:
    """Dataset load karo aur clean karo"""
    df = pd.read_csv(filepath, header=0 if has_header else None)
    if not has_header:
        df.columns = ['id', 'topic', 'sentiment', 'text']

    # Irrelevant rows hatao
    df = df[df['sentiment'] != 'Irrelevant'].copy()
    df.dropna(subset=['text'], inplace=True)

    # Clean text column add karo
    df['clean_text'] = df['text'].apply(clean_text)
    return df


if __name__ == '__main__':
    train = load_and_clean('data/twitter_training.csv')
    val   = load_and_clean('data/twitter_validation.csv')
    print(f"Train: {len(train)} rows | Val: {len(val)} rows")
    print(train['sentiment'].value_counts())
