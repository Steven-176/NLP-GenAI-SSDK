import spacy
import pandas as pd
import re
import html

# Charger le modèle de langue anglaise dans SpaCy
nlp = spacy.load("en_core_web_sm")

# Fichiers d'entrée
reviews_file = "reviews.jsonl"
meta_file = "meta.jsonl"

# Charger les données avec Pandas
reviews_df = pd.read_json(reviews_file, lines=True)
meta_df = pd.read_json(meta_file, lines=True)

# Fusion des champs "title" et "text" pour créer une colonne "document"
reviews_df["document"] = reviews_df["title"].fillna("") + " " + reviews_df["text"].fillna("")

# Fonction de nettoyage du texte brut
def clean_text(text):
    # Décoder les entités HTML
    text = html.unescape(text)
    # Supprimer les balises HTML
    text = re.sub(r'<[^>]+>', ' ', text)
    # Remplacer les contractions courantes
    contractions = {"don't": "do not", "can't": "cannot", "i'm": "i am", "you're": "you are", "it's": "it is"}
    for contraction, full_form in contractions.items():
        text = text.replace(contraction, full_form)
    # Supprimer les caractères non alphanumériques inutiles
    text = re.sub(r'[^\w\s]', ' ', text)
    # Supprimer les espaces multiples
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Nettoyer les documents
reviews_df["cleaned_document"] = reviews_df["document"].apply(clean_text)

# Fonction de prétraitement des textes avec SpaCy
def preprocess_text(doc):
    doc = nlp(doc)
    tokens = [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop  # Supprimer les stop words
        and not token.is_punct  # Supprimer la ponctuation
        and not token.like_num  # Supprimer les nombres
        and not token.is_space  # Supprimer les espaces
        and len(token) > 2  # Supprimer les mots courts (<= 2 caractères)
    ]
    # Supprimer les répétitions
    return list(dict.fromkeys(tokens))

# Appliquer le nettoyage et le prétraitement
reviews_df["processed"] = reviews_df["cleaned_document"].apply(preprocess_text)

# Sauvegarder les données prétraitées dans un fichier JSON
output_file = "preprocessed_reviews.jsonl"
reviews_df[["processed"]].to_json(output_file, orient="records", lines=True)

print(f"Données prétraitées sauvegardées dans {output_file}")
