import os
import pandas as pd
from tqdm import tqdm
from gensim.models import KeyedVectors
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer

# ---------------- CONFIG ----------------
TOP_WORDS = 2000
SIM_TOPN = 3
OUTPUT_DIR = "neo4j_import"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- LOAD WORD2VEC ----------------
print("Loading Word2Vec...")
model = KeyedVectors.load_word2vec_format(
    "GoogleNews-vectors-negative300.bin",
    binary=True
)

# ---------------- LOAD DATA ----------------
print("Loading dataset...")
data = fetch_20newsgroups(subset='train')

vectorizer = CountVectorizer(
    stop_words='english',
    max_features=TOP_WORDS
)

X = vectorizer.fit_transform(data.data)
words = vectorizer.get_feature_names_out()

# ---------------- WORD NODES ----------------
pd.DataFrame({"text": words}).to_csv(f"{OUTPUT_DIR}/words.csv", index=False)

# ---------------- CATEGORY NODES ----------------
categories = list(set(data.target_names))
pd.DataFrame({"name": categories}).to_csv(f"{OUTPUT_DIR}/categories.csv", index=False)

# ---------------- DOCUMENTS ----------------
docs = []
doc_cat = []
doc_word = []

print("Processing documents...")

for i, doc in enumerate(tqdm(data.data)):
    doc_id = f"DOC_{i}"
    category = data.target_names[data.target[i]]

    docs.append({"id": doc_id})
    doc_cat.append((doc_id, category))

    for word in doc.split():
        if word in words:
            doc_word.append((doc_id, word))

pd.DataFrame(docs).to_csv(f"{OUTPUT_DIR}/documents.csv", index=False)
pd.DataFrame(doc_cat, columns=["doc_id", "category"]).to_csv(f"{OUTPUT_DIR}/doc_category.csv", index=False)
pd.DataFrame(doc_word, columns=["doc_id", "word"]).to_csv(f"{OUTPUT_DIR}/doc_word.csv", index=False)

# ---------------- WORD SIMILARITY ----------------
print("Computing similarity...")

edges = []

for word in tqdm(words):
    if word in model:
        try:
            for sim_word, score in model.most_similar(word, topn=SIM_TOPN):
                edges.append((word, sim_word, score))
        except:
            pass

pd.DataFrame(edges, columns=["word1", "word2", "score"]).to_csv(
    f"{OUTPUT_DIR}/word_similarity.csv", index=False
)

print("DONE: CSV files created.")