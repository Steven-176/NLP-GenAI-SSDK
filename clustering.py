import json
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import matplotlib.pyplot as plt
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

# Charger les documents pré-traités
input_file = "pretreated_reviews.json"
with open(input_file, "r", encoding="utf-8") as f:
    documents = json.load(f)

# Étape 1 : Génération des embeddings
print("\n-------------------    Étape 1 : Génération des embeddings   -------------------\n")
model = SentenceTransformer('all-MiniLM-L12-v2')
embeddings = model.encode([" ".join(doc) for doc in documents])

# Étape 2 : Clustering
print("\n-------------------    Étape 2 : Clustering   -------------------\n")
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)
print(f"Centres des clusters :\n{kmeans.cluster_centers_}")

# Réduction de dimensions et visualisation
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)
plt.figure(figsize=(10, 8))
scatter = plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=cluster_labels, cmap='viridis', s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('Clustering des documents avec KMeans')
plt.xlabel('Dimension 1 (PCA)')
plt.ylabel('Dimension 2 (PCA)')
plt.show()

# Silhouette Score
silhouette = silhouette_score(embeddings, cluster_labels)
print(f"Silhouette Score: {silhouette}")

# Étape 3.1 : Association des documents aux clusters
print("\n-------------------    Étape 3.1 : Documents par cluster   -------------------\n")
documents_by_cluster = {i: [] for i in range(num_clusters)}
for doc, label in zip(documents, cluster_labels):
    documents_by_cluster[label].append(doc)

# Étape 3.2 : Fréquences des mots
print("\n-------------------    Étape 3.2 : Fréquences des mots   -------------------\n")
word_frequencies_by_cluster = {}
for cluster, docs in documents_by_cluster.items():
    all_words = [word for doc in docs for word in doc]
    word_counts = Counter(all_words)
    word_frequencies_by_cluster[cluster] = word_counts.most_common(10)

# Afficher les mots les plus fréquents par cluster
for cluster, common_words in word_frequencies_by_cluster.items():
    print(f"\nCluster {cluster} - Mots les plus fréquents :")
    for word, freq in common_words:
        print(f"{word}: {freq}")

# Étape 3.3 : Vérification de pertinence
print("\n-------------------    Étape 3.3 : Vérification de pertinence   -------------------\n")
for cluster, docs in documents_by_cluster.items():
    print(f"\nCluster {cluster} : {len(docs)} documents")
    print("Exemple de mots fréquents :", [word for word, freq in word_frequencies_by_cluster[cluster]])
    print("Exemple de document :", " ".join(docs[0][:20]) if docs else "Aucun document")

# Étape 4 : Extraction des mots-clés avec TF-IDF
print("\n-------------------    Étape 4 : Extraction des mots-clés (TF-IDF)   -------------------\n")

for cluster, docs in documents_by_cluster.items():
    # Préparer les textes pour TF-IDF
    all_text = [" ".join(doc) for doc in docs]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    
    # Obtenir les mots et leurs scores
    feature_names = vectorizer.get_feature_names_out()
    tfidf_scores = tfidf_matrix.sum(axis=0).A1  # Somme des scores TF-IDF pour chaque mot
    sorted_items = sorted(zip(tfidf_scores, feature_names), reverse=True)  # Trier par pertinence
    
    # Afficher les 10 mots-clés les plus importants
    print(f"\nCluster {cluster} - Mots-clés TF-IDF :")
    for score, word in sorted_items[:10]:
        print(f"{word}: {score:.2f}")
