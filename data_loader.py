import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(url):
    df = pd.read_csv(url, sep='\t', header=None, names=['label', 'message'])
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)
