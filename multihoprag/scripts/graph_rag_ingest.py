from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# =========================
# CONFIG
# =========================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "!I10wrk01"
DATABASE = "cs564multihoprag"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 25  # 🔥 batching is key for performance

# =========================
# LOAD MODEL
# =========================

model = SentenceTransformer(EMBED_MODEL_NAME)

# =========================
# DRIVER
# =========================

driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD)
)

# =========================
# CHUNKING (STREAM SAFE)
# =========================

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    start = 0

    while start < len(words):
        yield " ".join(words[start:start + chunk_size])
        start += chunk_size - overlap


# =========================
# EMBEDDINGS
# =========================

def get_embedding(text: str):
    vec = model.encode(text).tolist()
    assert len(vec) == 384
    return vec


# =========================
# NEO4J OPERATIONS
# =========================

def create_document(tx, doc_id):
    tx.run("""
    MERGE (d:Document {doc_id: $doc_id})
    """, doc_id=doc_id)


def create_batch(tx, doc_id, batch):
    tx.run("""
    MATCH (d:Document {doc_id: $doc_id})

    UNWIND $batch AS item

    CREATE (c:Chunk {
        chunk_id: item.chunk_id,
        text: item.text,
        embedding: item.embedding
    })

    CREATE (d)-[:HAS_CHUNK]->(c)
    """,
    doc_id=doc_id,
    batch=batch)


# =========================
# INGESTION PIPELINE
# =========================

def ingest_document(doc_id, text):
    print(f"\n📦 Ingesting document: {doc_id}")

    session = driver.session(database=DATABASE)

    try:
        # 1. Create document
        session.execute_write(create_document, doc_id)

        # 2. Stream chunks (NO len())
        chunk_stream = chunk_text(text)

        batch = []
        chunk_counter = 0

        for chunk in tqdm(chunk_stream, desc="Processing chunks"):

            chunk_id = f"{doc_id}_chunk_{chunk_counter}"
            embedding = get_embedding(chunk)

            batch.append({
                "chunk_id": chunk_id,
                "text": chunk,
                "embedding": embedding
            })

            chunk_counter += 1

            # 3. Batch write
            if len(batch) >= BATCH_SIZE:
                session.execute_write(create_batch, doc_id, batch)
                batch.clear()

        # flush remaining
        if batch:
            session.execute_write(create_batch, doc_id, batch)

        print(f"\n✅ Ingestion complete: {chunk_counter} chunks")

    finally:
        session.close()


# =========================
# VECTOR TEST
# =========================

def test_vector_search(query, top_k=5):
    embedding = get_embedding(query)

    with driver.session(database=DATABASE) as session:
        result = session.run("""
        CALL db.index.vector.queryNodes(
            'chunk_embedding_index',
            $top_k,
            $embedding
        )
        YIELD node, score
        RETURN node.text AS text, score
        ORDER BY score DESC
        """,
        top_k=top_k,
        embedding=embedding)

        return [r.data() for r in result]


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    sample_text = """
    Graph databases store relationships between entities.
    Neo4j is a popular graph database used for knowledge graphs and retrieval augmented generation.
    GraphRAG combines vector search with graph traversal to improve LLM retrieval accuracy.
    """

    ingest_document("doc_001", sample_text)

    print("\n🔎 Testing vector search...\n")

    results = test_vector_search("What is GraphRAG?", top_k=3)

    for r in results:
        print(r)