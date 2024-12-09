import pandas as pd
import spacy
import re
import json

def clean_text_before_tokenization(doc):

    doc = re.sub(r"<.*?>", " ", doc)

    doc = re.sub(r"\b\w{1,2}\b", " ", doc)

    doc = re.sub(r"\s+", " ", doc).strip()
    return doc

reviews_file_path = './reviews.jsonl'

reviews_df = pd.read_json(reviews_file_path, lines=True)

selected_fields = reviews_df[['title', 'text']].dropna()

documents = (selected_fields['title'] + " " + selected_fields['text']).tolist()

nlp = spacy.load("en_core_web_sm")

documents = [clean_text_before_tokenization(doc) for doc in documents]

lemmatized_documents = [[token.lemma_.lower() for token in nlp(doc)] for doc in documents]


print("\n---------------------- 1 ----------------------\n")

filtered_tokens = [
    [
        token.text for token in nlp(" ".join(doc)) 
        if not token.is_stop and
        not token.is_punct and
        len(token) > 2 and
        not token.is_digit and
        not token.like_url and
        not token.like_email and
        token.is_alpha
    ] 
    for doc in lemmatized_documents]

print(filtered_tokens[:5])

print("\n---------------------- 2 ----------------------\n")

output_file = "pretreated_reviews.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(filtered_tokens, f, ensure_ascii=False)

print(f"Données sauvegardées dans le fichier : {output_file}")