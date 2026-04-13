import os
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from tqdm import tqdm

# ---------------- CONFIG ----------------
URI = "bolt://localhost:7687"
# AUTH = ("CS564FinalProject", "!I10wrk01")
AUTH = ("neo4j", "!I10wrk01")

DB_NAME = "cs564multihoprag"

EMBED_MODEL = "all-MiniLM-L6-v2"

driver = GraphDatabase.driver(URI, auth=AUTH)
embedder = SentenceTransformer(EMBED_MODEL)

# ---------------- LOAD DATASET ----------------
dataset = load_dataset(
    "yixuantt/MultiHopRAG",
    "MultiHopRAG",
    split="train",
    streaming=False
)

print(f"Dataset loaded: {len(dataset)} examples")

# ---------------- NEO4J FUNCTIONS ----------------

def create_question(tx, qid, text):
    tx.run("""
    MERGE (q:Question {id:$id})
    SET q.text = $text
    """, id=qid, text=text)


def create_chunk(tx, cid, text, embedding):
    tx.run("""
    MERGE (c:Chunk {id:$id})
    SET c.text = $text,
        c.embedding = $embedding
    """, id=cid, text=text, embedding=embedding)


def link_q_chunk(tx, qid, cid):
    tx.run("""
    MATCH (q:Question {id:$q}), (c:Chunk {id:$c})
    MERGE (q)-[:HAS_CHUNK]->(c)
    """, q=qid, c=cid)

# ---------------- EMBEDDING ----------------

def embed_texts(texts):
    return embedder.encode(texts, show_progress_bar=False).tolist()

# ---------------- FIELD SAFE ACCESS ----------------

def get_question(item):
    return (
        item.get("question")
        or item.get("query")
        or item.get("input")
        or ""
    )

def get_contexts(item):
    ctx = (
        item.get("context")
        or item.get("passages")
        or []
    )
    return ctx if isinstance(ctx, list) else [ctx]

# ---------------- MAIN PIPELINE ----------------

print(f"Building Microsoft-style GraphRAG → {DB_NAME}")

with driver.session(database=DB_NAME) as session:

    for i, item in enumerate(tqdm(dataset)):

        # ---------- QUESTION ----------
        qid = f"Q{i}"
        question = get_question(item)

        session.execute_write(create_question, qid, question)

        # ---------- CONTEXTS ----------
        contexts = get_contexts(item)

        # clean + filter
        chunks = [c.strip() for c in contexts if isinstance(c, str) and c.strip()]

        if not chunks:
            continue

        # ---------- EMBEDDINGS (BATCHED) ----------
        embeddings = embed_texts(chunks)

        # ---------- STORE GRAPH ----------
        for j, (chunk, emb) in enumerate(zip(chunks, embeddings)):

            cid = f"{qid}_C{j}"

            session.execute_write(create_chunk, cid, chunk, emb)
            session.execute_write(link_q_chunk, qid, cid)

print("DONE: Production GraphRAG build complete")

driver.close()