import pandas as pd

meta = pd.read_json('meta.jsonl', lines=True)
reviews = pd.read_json('reviews.jsonl', lines=True)

print(reviews.head())