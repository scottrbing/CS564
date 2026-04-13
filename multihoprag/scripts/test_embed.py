import os
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---------------- CONFIG ----------------
URI = "bolt://localhost:7687"
AUTH = ("neo4j", "!I10wrk01")

DB_NAME = "cs564multihoprag"

EMBED_MODEL = "all-MiniLM-L6-v2"

driver = GraphDatabase.driver(URI, auth=AUTH)
embedder = SentenceTransformer(EMBED_MODEL)

embedding = embedder.encode("Hello, my dog is cute")
print("embedding shape:", len(embedding))