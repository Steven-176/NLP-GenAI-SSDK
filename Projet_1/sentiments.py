import json
from transformers import pipeline
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import pandas as pd

# Étape 1 : Chargement des avis
print("Chargement des avis...")
with open("reviews.jsonl", "r", encoding="utf-8") as f:
    reviews_data = [json.loads(line) for line in f]

# Filtrer les avis contenant une note et un texte
data = [
    {"text": review["text"], "rating": review["rating"]}
    for review in reviews_data
    if "rating" in review and "text" in review
]

# Étape 2 : Chargement du pipeline Hugging Face pour l'analyse des sentiments
print("Chargement du pipeline de sentiment...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="nlptown/bert-base-multilingual-uncased-sentiment",
    truncation=True
)

# Étape 3 : Analyse des sentiments
print("Analyse des sentiments...")
true_ratings, predicted_ratings = [], []

for entry in data:
    result = sentiment_pipeline(entry["text"])[0]
    pred = int(result["label"].split()[0])
    predicted_ratings.append(pred)
    true_ratings.append(entry["rating"])

# Vérifier si les données sont suffisantes pour la corrélation
if len(true_ratings) < 2 or len(predicted_ratings) < 2:
    print("Pas assez de données pour calculer la corrélation.")
    exit()

# Étape 4 : Évaluation des performances
print("Évaluation des performances...")
correlation, _ = pearsonr(true_ratings, predicted_ratings)
print(f"Corrélation de Pearson : {correlation:.4f}")

# Étape 5 : Visualisation des résultats
print("Visualisation des résultats...")

# Créer un DataFrame pour faciliter l'analyse
df = pd.DataFrame({
    "True Ratings": true_ratings,
    "Predicted Ratings": predicted_ratings
})

# Visualisation des distributions
plt.figure(figsize=(12, 6))

# Distribution des notes réelles
plt.subplot(1, 2, 1)
df["True Ratings"].value_counts(sort=False).plot(kind="bar", color="blue", alpha=0.7)
plt.title("Distribution des notes réelles")
plt.xlabel("Notes")
plt.ylabel("Nombre d'avis")
plt.xticks(range(1, 6))

# Distribution des notes prédites
plt.subplot(1, 2, 2)
df["Predicted Ratings"].value_counts(sort=False).plot(kind="bar", color="green", alpha=0.7)
plt.title("Distribution des notes prédites")
plt.xlabel("Notes")
plt.ylabel("Nombre d'avis")
plt.xticks(range(1, 6))

# Afficher les graphiques
plt.tight_layout()
plt.show()