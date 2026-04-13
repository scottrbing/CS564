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

#client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key="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")

# ---------------- EMBEDDING ----------------
def embed(text):
    return embedder.encode(text).tolist()

# ---------------- VECTOR SEARCH ----------------
def vector_search(tx, query_embedding, k=5):
    query = """
    CALL db.index.vector.queryNodes('chunk_embedding_index', $k, $embedding)
    YIELD node, score
    RETURN node.id AS id, node.text AS text, score
    ORDER BY score DESC
    """
    return list(tx.run(query, embedding=query_embedding, k=k))

# ---------------- MULTI-HOP EXPANSION ----------------
def expand_graph(tx, chunk_ids):
    query = """
    MATCH (q:Question)-[:HAS_CHUNK]->(c:Chunk)
    WHERE c.id IN $ids
    OPTIONAL MATCH (q)-[:HAS_CHUNK]->(c2:Chunk)
    RETURN DISTINCT c.text AS context, q.text AS question
    LIMIT 10
    """
    return list(tx.run(query, ids=chunk_ids))

# ---------------- BUILD CONTEXT ----------------
def build_context(results, expanded):
    context = "\n\n=== VECTOR MATCHES ===\n"
    for r in results:
        context += f"- {r['text']}\n"

    context += "\n\n=== MULTI-HOP EXPANSION ===\n"
    for r in expanded:
        context += f"Q: {r['question']}\nA-context: {r['context']}\n"

    return context

# ---------------- MAIN RETRIEVAL ----------------
def ask(question):
    q_emb = embed(question)

    with driver.session(database=DB_NAME) as session:

        # 1. Vector search
        results = session.execute_read(
            vector_search,
            q_emb,
            5
        )

        chunk_ids = [r["id"] for r in results]

        # 2. Graph expansion (multi-hop)
        expanded = session.execute_read(
            expand_graph,
            chunk_ids
        )

    # 3. Build prompt context
    context = build_context(results, expanded)

    prompt = f"""
You are a GraphRAG assistant.

Use the structured context below to answer the question.

--- CONTEXT ---
{context}

--- QUESTION ---
{question}

Instructions:
- Use BOTH vector matches and graph expansions
- Prefer graph-supported facts
- If unsure, say so clearly
"""

    # 4. LLM call
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content

# ---------------- CLI ----------------
def main():
    print("GraphRAG Retrieval Engine Ready (type 'exit')")

    while True:
        q = input("\nYou: ")
        if q.lower() in ["exit", "quit"]:
            break

        answer = ask(q)
        print("\nAI:", answer)

    driver.close()

if __name__ == "__main__":
    main()