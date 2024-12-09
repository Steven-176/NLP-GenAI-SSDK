# Projet d'Analyse des Données et NLP

Ce projet contient plusieurs scripts Python et notebooks Jupyter pour effectuer des analyses, du clustering et du traitement de langage naturel (NLP). Voici les étapes pour configurer et lancer le projet.

---

## **Prérequis**

Avant de commencer, assurez-vous d'avoir installé :
- **Python 3.8+**
- **pip**
- **virtualenv** (facultatif)

---

## **Installation**

### 1 Clonez le dépôt
```bash
git clone <URL_DU_DEPOT>
cd <NOM_DU_PROJET>
```

### 2 Créez et activez un environnement virtuel
**Sous Windows :**
```bash
python -m venv venv
venv\Scripts\activate
```

**Sous macOS/Linux :**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3 Installez les dépendances
Une fois l'environnement activé, installez les dépendances :
```bash
pip install -r requirements.txt
```

---

## **Utilisation**

### Lancer les scripts Python
Pour exécuter les différents scripts Python, utilisez la commande suivante :
```bash
python <nom_du_script>.py
```

Exemples :
```bash
python pretraitement.py
python clustering.py
python sentiments.py
```

### Lancer les notebooks Jupyter
1. Assurez-vous que Jupyter est installé :
   ```bash
   pip install notebook
   ```
2. Démarrez le serveur Jupyter :
   ```bash
   jupyter notebook
   ```
3. Ouvrez le fichier `rag.ipynb` dans votre navigateur et exécutez les cellules.

---

## **Structure du projet**

- **Projet 1**
  - `pretraitement.py` : Script pour nettoyer et préparer les données textuelles.
  - `clustering.py` : Script pour effectuer un clustering des données textuelles.
  - `sentiments.py` : Script pour analyser les sentiments des avis clients.

- **Projet 2**
  - `rag.ipynb` : Notebook permettant de créer un RAG (Retrieval-Augmented Generation) sur les descriptions des produits Amazon.