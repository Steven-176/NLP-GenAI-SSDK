import json
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"

input_file = "pretreated_reviews.json"

with open(input_file, "r", encoding="utf-8") as f:
    documents = json.load(f)

print("\n-------------------    1   -------------------\n")

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode([" ".join(doc) for doc in documents])

print("\n-------------------    2   -------------------\n")

# dbscan = DBSCAN(eps=0.7, min_samples=10)

# clusters = dbscan.fit_predict(embeddings)

# print("\n-------------------    3   -------------------\n")

# filtr_embeddings = embeddings[clusters != -1]
# filtr_clusters = clusters[clusters != -1]

# if len(set(clusters)) > 1:
#     silhouette_avg = silhouette_score(embeddings, clusters)
#     print(f"Silhouette Score moyen : {silhouette_avg}")
# else:
#     print("Impossible de calculer le score de silhouette : tous les points appartiennent au même cluster ou sont du bruit.")

from itertools import product

eps_values = [1.2, 1.5, 2.0]
min_samples_values = [10, 15]

for eps, min_samples in product(eps_values, min_samples_values):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clusters = dbscan.fit_predict(embeddings)

    nbr_clusters = set(clusters)

    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # Réduire les dimensions
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)

    # Visualiser les clusters
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=clusters, cmap='viridis', s=10)
    plt.title("Clusters DBSCAN (PCA)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(label='Cluster ID')
    plt.show()

    
    if len(set(clusters)) > 1:
        silhouette_avg = silhouette_score(embeddings, clusters)
        print(f"nombre de clusters={len(nbr_clusters)} -- eps={eps}, min_samples={min_samples}, silhouette score={silhouette_avg}")