import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
# from langchain.chains import RetrievalQA
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 1. Chargement des données
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    # Extraire les descriptions
    texts = []
    for item in data:
        if 'description' in item:
            description = item['description']
            if isinstance(description, list):  # Si c'est une liste, concaténer les éléments
                description = ' '.join(description)
            if isinstance(description, str) and description.strip():  # Vérifier que ce n'est pas vide
                texts.append(description.strip())
    return texts

# Charger les descriptions des produits
data = load_data('meta.jsonl')

# print(data)

# print(f"Nombre de descriptions chargées : {len(data)}")
# for i, description in enumerate(data[:5]):  # Afficher les 5 premières descriptions
#     print(f"Description {i+1}: {description}")

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

# for i,chunk in enumerate(chunks):
#     print(f'Description {i} : {chunks[i]}')

# 3. Génération des embeddings
embedding_model = HuggingFaceEmbeddings(model_name=MODEL_NAME)

# # Vérification des embeddings
# for i, chunk in enumerate(chunks[:5]):  # Limiter l'affichage aux 5 premiers chunks
#     embedding = embedding_model.embed_query(chunk)  # Générer l'embedding pour un chunk
#     print(f"Chunk {i + 1}: {chunk[:50]}...")  # Afficher un aperçu du chunk
#     print(f"Embedding {i + 1}: {embedding[:10]}...")  # Afficher les 10 premières valeurs de l'embedding
#     print(f"Taille de l'embedding : {len(embedding)}")

# 4. Créer la base vectorielle avec FAISS
vectorstore = FAISS.from_texts(
    texts=chunks,
    embedding=embedding_model
)

# 5. Configuration du système de récupération (retriever)
retriever = vectorstore.as_retriever()

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# 6. Configurer le modèle LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

retrieval_chain.invoke({"input": "Combien d'avis ont le mot 'great'"})
