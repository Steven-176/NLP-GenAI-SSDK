import openai
import json
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# 1. Chargement des données
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

# Charger les descriptions des produits
data = load_data('meta.jsonl')

# Extraire les descriptions
texts = [item.get('description', '') for item in data if 'description' in item]

# 2. Prétraitement et segmentation
def preprocess_texts(texts, chunk_size=512, chunk_overlap=128):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks

chunks = preprocess_texts(texts)

# 3. Génération des embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'))

# 4. Création de la base vectorielle
vectorstore = Chroma.from_texts(chunks, embedding_model)

# 5. Configuration du système de récupération (retriever)
retriever = vectorstore.as_retriever()

# 6. Conception de la chaîne RAG
rag_model = ChatOpenAI(model_name="gpt-4", openai_api_key=os.getenv('OPENAI_API_KEY'))
rag_chain = RetrievalQA(combine_documents_chain=rag_model, retriever=retriever)

# 7. Exécution de requêtes utilisateur
def ask_question(question):
    result = rag_chain.run(question)
    return result

# Exemple de requête utilisateur
if __name__ == "__main__":
    print("Posez une question : ")
    user_question = input()
    response = ask_question(user_question)
    print("\nRéponse générée :")
    print(response)
