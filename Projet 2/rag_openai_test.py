import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Chargement des données
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # Extraire les descriptions
    texts = [item.get('description', '') for item in data if 'description' in item]
    return texts

# Charger les descriptions des produits
data = load_data('meta.jsonl')

# print(data)

print(f"Nombre de descriptions chargées : {len(data)}")
for i, description in enumerate(data[:5]):  # Afficher les 5 premières descriptions
    print(f"Description {i+1}: {description}")

# 2. Prétraitement et segmentation
def preprocess_texts(texts, chunk_size=512, chunk_overlap=128):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = []
    for text in texts:
        if isinstance(text, str): # Vérifier que la description est une chaîne
            chunks.extend(splitter.split_text(text))
    return chunks

chunks = preprocess_texts(data)

print(chunks)