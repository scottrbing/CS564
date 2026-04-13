import streamlit as st
from neo4j import GraphDatabase
import pandas as pd

# -----------------------------
# Neo4j Connection
# -----------------------------
class Neo4jClient:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    # -----------------------------
    # Vector Search (Neo4j Vector Index)
    # -----------------------------
    def vector_search(self, query_embedding, top_k=5):
        cypher = """
        CALL db.index.vector.queryNodes('chunk_embedding_index', $top_k, $embedding)
        YIELD node, score
        MATCH (d:Document)-[:HAS_CHUNK]->(node)
        RETURN d.id AS document, node.text AS chunk, score
        ORDER BY score DESC
        LIMIT $top_k
        """
        with self.driver.session() as session:
            results = session.run(
                cypher,
                embedding=query_embedding,
                top_k=top_k
            )
            return [r.data() for r in results]

    # -----------------------------
    # Graph Expansion from Chunks
    # -----------------------------
    def graph_expand(self, chunk_texts):
        cypher = """
        UNWIND $chunks AS chunk_text
        MATCH (c:Chunk)
        WHERE c.text = chunk_text
        MATCH (d:Document)-[:HAS_CHUNK]->(c)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e:Entity)
        OPTIONAL MATCH (c)-[:MENTIONS]->(e2:Entity)<-[:MENTIONS]-(c2:Chunk)
        RETURN d.id AS document,
               c.text AS chunk,
               collect(DISTINCT e.name) AS direct_entities,
               collect(DISTINCT c2.text) AS related_chunks
        LIMIT 50
        """
        with self.driver.session() as session:
            results = session.run(cypher, chunks=chunk_texts)
            return [r.data() for r in results]

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="GraphRAG Side-by-Side", layout="wide")

st.title("🧠 GraphRAG Side-by-Side Explorer")
st.markdown("Compare Vector Retrieval vs Graph Expansion")

# Sidebar config
st.sidebar.header("⚙️ Neo4j Config")
uri = st.sidebar.text_input("Neo4j URI", "bolt://localhost:7687")
user = st.sidebar.text_input("Username", "neo4j")
password = st.sidebar.text_input("Password", type="password")

query = st.text_input("Enter query", "GraphRAG")

# Placeholder embedding function (replace with real embedder)
def embed_text(text):
    import numpy as np
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(1536).tolist()

if st.button("Run Comparison"):
    if not password:
        st.error("Missing Neo4j password")
    else:
        client = Neo4jClient(uri, user, password)

        # -----------------------------
        # VECTOR RETRIEVAL
        # -----------------------------
        embedding = embed_text(query)
        vector_results = client.vector_search(embedding, top_k=5)

        vector_chunks = [r["chunk"] for r in vector_results]

        # -----------------------------
        # GRAPH EXPANSION
        # -----------------------------
        graph_results = client.graph_expand(vector_chunks)

        client.close()

        # -----------------------------
        # SIDE BY SIDE UI
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Vector Retrieval")
            if vector_results:
                df_vec = pd.DataFrame(vector_results)
                st.dataframe(df_vec, use_container_width=True)
            else:
                st.warning("No vector results found")

        with col2:
            st.subheader("🕸️ Graph Expansion")
            if graph_results:
                df_graph = pd.DataFrame(graph_results)
                st.dataframe(df_graph, use_container_width=True)

                st.markdown("### Entity & Context Summary")
                for row in graph_results:
                    st.markdown(f"**Doc:** {row['document']}")
                    st.markdown(f"Chunk: {row['chunk']}")
                    st.markdown(f"Entities: {row['direct_entities']}")
                    st.markdown(f"Related chunks: {len(row['related_chunks'])}")
                    st.markdown("---")
            else:
                st.warning("No graph expansion results found")

# Footer
st.markdown("---")
st.caption("GraphRAG Side-by-Side Visualizer | Vector vs Graph Retrieval")
