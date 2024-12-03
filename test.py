import pandas as pd
import spacy
import re

# Charger le modèle SpaCy en anglais
nlp = spacy.load("en_core_web_sm")

# Charger les données des avis dans un DataFrame pandas
df = pd.read_json('./reviews.json')

# Sélectionner les colonnes pertinentes
df = df[['title', 'text']].dropna()  # Supprimer les lignes avec des valeurs manquantes

# Fonction pour nettoyer le texte brut
def clean_text(text):
    # Supprimer les balises HTML et les caractères non pertinents
    text = re.sub(r"<[^>]*>", " ", text)  # Balises HTML
    text = re.sub(r"\\u[0-9a-fA-F]{4}", " ", text)  # Séquences unicode comme \u263a
    text = re.sub(r"[^\w\s]", " ", text)  # Supprimer la ponctuation
    text = re.sub(r"\s+", " ", text)  # Remplacer les espaces multiples par un seul espace
    return text.strip()

# Fonction de prétraitement des textes avec SpaCy
def preprocess_text(text):
    doc = nlp(text)
    tokens = [
        token.lemma_.lower() for token in doc
        if not token.is_stop and not token.is_punct and not token.is_digit and token.is_alpha
    ]
    return tokens  # Retourner une liste de tokens nettoyés

# Appliquer le nettoyage et le prétraitement aux colonnes "title" et "text"
df['processed_title'] = df['title'].apply(lambda x: preprocess_text(clean_text(x)))
df['processed_text'] = df['text'].apply(lambda x: preprocess_text(clean_text(x)))

# Créer une liste de documents, chaque document correspondant à un avis
documents = df.apply(
    lambda row: row['processed_title'] + row['processed_text'], axis=1
).tolist()

# Sauvegarder la liste des documents dans un fichier JSON
output_file = './processed_reviews.json'
pd.DataFrame({'document': documents}).to_json(output_file, orient='records', lines=True)

print(f"Fichier sauvegardé en JSON : {output_file}")