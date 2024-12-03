import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Charger les données prétraitées
preprocessed_file = "preprocessed_reviews.json"
df = pd.read_json(preprocessed_file, lines=True)

# Vérifier les colonnes disponibles
print("Colonnes disponibles :", df.columns)

# Vérifier si la colonne contenant les tokens est correcte
if "processed" in df.columns:
    df["document"] = df["processed"].apply(" ".join)
else:
    raise ValueError("La colonne 'processed' n'existe pas dans le fichier JSONL chargé.")

# Étape 1 : Génération des embeddings avec TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # Limite les features pour éviter la surcharge
tfidf_matrix = tfidf_vectorizer.fit_transform(df["document"])

# Étape 2 : Clustering avec KMeans
num_clusters = 5  # Choix arbitraire, ajustable
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df["cluster"] = kmeans.fit_predict(tfidf_matrix)

# Étape 3 : Analyse des clusters
# Identifiez les mots-clés pour chaque cluster
def extract_keywords(tfidf_matrix, clusters, feature_names, num_keywords=10):
    keywords = {}
    for cluster in range(num_clusters):
        cluster_indices = (df["cluster"] == cluster)
        cluster_matrix = tfidf_matrix[cluster_indices]
        mean_tfidf = cluster_matrix.mean(axis=0).A1
        top_indices = mean_tfidf.argsort()[-num_keywords:][::-1]
        keywords[cluster] = [feature_names[i] for i in top_indices]
    return keywords

# Extraire les mots-clés
feature_names = tfidf_vectorizer.get_feature_names_out()
keywords_per_cluster = extract_keywords(tfidf_matrix, df["cluster"], feature_names)

# Étape 4 : Calcul du Silhouette Score
silhouette_avg = silhouette_score(tfidf_matrix, df["cluster"])
print(f"Silhouette Score : {silhouette_avg:.3f}")

# Sauvegarde des résultats
output_file = "clustering_results.json"
results = {
    "silhouette_score": silhouette_avg,
    "keywords_per_cluster": keywords_per_cluster,
    "documents_per_cluster": df.groupby("cluster")["document"].apply(list).to_dict(),
}
pd.DataFrame(results).to_json(output_file, orient="records", lines=True)

print(f"Clustering terminé. Résultats sauvegardés dans {output_file}")