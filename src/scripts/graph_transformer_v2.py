"""
Knowledge Graph Builder v2
===========================
Builds a high-quality Neo4j graph from the evidence corpus.

Key improvements over v1:
  1. gpt-4o for extraction (much better relationship typing)
  2. Metadata injected into chunk text (source, title, date) 
     so extractor can create REPORTED_BY edges
  3. No catch-all edges — every relationship is specific
  4. Schema descriptions passed to LLM via additional_instructions
  5. Post-ingestion: Publication edge creation from metadata (safety net)
  6. Post-ingestion: Entity deduplication

Usage:
    # First run: python prepare_evidence_corpus.py
    # Then clear Neo4j: MATCH (n) DETACH DELETE n
    # Then: python graph_transformer_v2.py
"""

import time
import json
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
from src.db.graph_store import GraphStoreManager
from src.configs.config import settings


# Schema Loading — now loads descriptions too since this was important

def get_nodes_and_edges(path):
    with open(path, 'r') as f:
        data = json.load(f)
    nodes = [n['label'] for n in data['nodes']]
    edges = [e['label'] for e in data['edges']]
    return nodes, edges


def build_additional_instructions(schema_path):
    """
    Build a detailed instruction string from schema descriptions.
    This gets passed to LLMGraphTransformer's additional_instructions parameter
    so gpt-4o actually SEES the node/edge definitions during extraction.
    """
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    lines = []

    lines.append("## NODE TYPE DEFINITIONS — use these EXACTLY:")
    for node in schema['nodes']:
        lines.append(f"- {node['label']}: {node['description']}")

    lines.append("\n## RELATIONSHIP DEFINITIONS — use these EXACTLY:")
    for edge in schema['edges']:
        lines.append(f"- {edge['label']}: {edge['description']}")

    lines.append("\n## CRITICAL RULES:")
    lines.append("1. REPORTED_BY is the MOST IMPORTANT edge. Every chunk starts with a [Source: X] prefix — ALWAYS create (Entity)-[REPORTED_BY]->(Publication) edges from this.")
    lines.append("2. NEVER invent relationship types not listed above. If no listed type fits, skip that relationship.")
    lines.append("3. Use CEO_OF for CEO/chief executive roles. Use WORKS_FOR for all other employment including athletes playing for teams.")
    lines.append("4. Use FOUNDED for entity creation, OWNS for acquisitions/ownership, INVESTED_IN for financial investments — these are distinct.")
    lines.append("5. Extract the Publication name from the [Source: X] metadata at the start of each chunk and create a Publication node for it.")

    return "\n".join(lines)


# Chunking with Metadata Injection

def get_chunked_dataset(df):
    """
    Chunk articles with metadata INJECTED into the text.
    
    The LLM extractor only sees page_content — if source/date aren't 
    in the text, it can never extract REPORTED_BY edges.
    Every chunk starts with: [Source: X | Title: Y | Published: Z]
    """
    df['doc_id'] = [f"doc_{i}" for i in range(len(df))]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=150
    )

    chunked_docs = []

    for _, row in df.iterrows():
        raw_text = row.get('body', '')
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        title = str(row.get('title', '')).strip()
        source = str(row.get('source', '')).strip()
        published_at = str(row.get('published_at', '')).strip()
        
        # Clean date
        if 'T' in published_at:
            published_at = published_at.split('T')[0]

        # Build prefix
        meta_parts = []
        if source:
            meta_parts.append(f"Source: {source}")
        if title:
            meta_parts.append(f"Title: {title}")
        if published_at:
            meta_parts.append(f"Published: {published_at}")

        prefix = f"[{' | '.join(meta_parts)}]\n\n" if meta_parts else ""

        chunks = text_splitter.split_text(raw_text)

        for i, chunk_text in enumerate(chunks):
            doc = Document(
                page_content=prefix + chunk_text,
                metadata={
                    "doc_id": row['doc_id'],
                    "chunk_id": f"{row['doc_id']}_chunk_{i}",
                    "title": title,
                    "source": source,
                    "published_at": published_at
                }
            )
            chunked_docs.append(doc)

    print(f"Generated {len(chunked_docs)} metadata-enriched chunks")
    return chunked_docs


# Graph Builder — now with additional_instructions which is the description field of the schema.

class KnowledgeGraphBuilder:
    def __init__(self, schema_path, model="gpt-4o"):
        self.graph_db = GraphStoreManager.get_neo4j_graph()

        print(f"LLM model: {model}")
        self.llm = ChatOpenAI(
            temperature=0,
            model=model,
            api_key=settings.OPENAI_API_KEY
        )

        nodes, edges = get_nodes_and_edges(schema_path)
        self.allowed_nodes = nodes
        self.allowed_edges = edges

        # Build the instruction string with all descriptions
        instructions = build_additional_instructions(schema_path)

        print(f"Schema: {len(nodes)} node types, {len(edges)} edge types")
        print(f"  Nodes: {nodes}")
        print(f"  Edges: {edges}")
        print(f"  Additional instructions: {len(instructions)} chars (~{len(instructions)//4} tokens)")

        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=self.allowed_nodes,
            allowed_relationships=self.allowed_edges,
            strict_mode=True,
            node_properties=False,
            relationship_properties=False,
            additional_instructions=instructions,
        )


# Batch Ingestion, so that if it fails, we can retry that later
def build_graph_with_fallback(
    chunked_documents,
    builder: KnowledgeGraphBuilder,
    batch_size=10,
    error_file="failed_chunks_v2.json"
):
    total = len(chunked_documents)
    print(f"Processing {total} chunks (batch_size={batch_size})")

    failed_chunks = []
    total_triples = 0

    for i in tqdm(range(0, total, batch_size), desc="Extracting"):
        batch = chunked_documents[i: i + batch_size]

        try:
            graph_docs = builder.transformer.convert_to_graph_documents(batch)
            batch_triples = sum(len(gd.relationships) for gd in graph_docs)
            total_triples += batch_triples

            builder.graph_db.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True
            )

            time.sleep(5.0)  # Rate limit for gpt-4o

        except Exception as e:
            print(f"\n[ERROR] Batch {i}-{i+len(batch)}: {str(e)[:100]}")
            for doc in batch:
                failed_chunks.append({
                    "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                    "title": doc.metadata.get("title", "unknown"),
                    "error": str(e)[:200]
                })
            continue

    print(f"\nExtracted ~{total_triples} triples total")

    if failed_chunks:
        print(f"[WARN] {len(failed_chunks)} chunks failed")
        with open(error_file, 'w') as f:
            json.dump(failed_chunks, f, indent=2)
    else:
        print("[SUCCESS] All chunks processed")

    return failed_chunks


# Post-Ingestion: Publication Edge Safety since a lot of the query mentions the publication article
def create_publication_edges(graph, df):
    """
    Create REPORTED_BY edges from metadata as a safety net.
    Even if the LLM misses these during extraction, we guarantee
    every Document node connects to its source Publication.
    """
    print("\n[POST] Creating Publication nodes and REPORTED_BY edges...")

    sources = df['source'].dropna().unique()
    for source in sources:
        try:
            graph.query(
                "MERGE (p:Publication:__Entity__ {id: $source})",
                params={"source": source}
            )
        except Exception as e:
            print(f"  Failed to create Publication '{source}': {e}")

    print(f"  Created {len(sources)} Publication nodes")

    # Link Documents to Publications
    linked = 0
    for _, row in df.iterrows():
        source = row.get('source', '')
        title = str(row.get('title', ''))
        if not source or not title:
            continue

        try:
            result = graph.query(
                """
                MATCH (d:Document)
                WHERE d.text CONTAINS $title_fragment
                MATCH (p:Publication {id: $source})
                MERGE (d)-[:REPORTED_BY]->(p)
                RETURN count(*) AS cnt
                """,
                params={"title_fragment": title[:50], "source": source}
            )
            if result and result[0]["cnt"] > 0:
                linked += result[0]["cnt"]
        except:
            pass

    print(f"  Linked {linked} Document->Publication edges")


# Post-Ingestion: Entity Deduplication
def run_deduplication(graph):
    """
    Find entities where one name contains another and merge them.
    E.g., 'Sam Bankman-Fried' and 'Bankman-Fried' -> keep longer one.
    """
    print("\n[POST] Deduplicating entities...")

    FIND_DUPES = """
    MATCH (a), (b)
    WHERE a.id <> b.id
      AND id(a) < id(b)
      AND (toLower(a.id) CONTAINS toLower(b.id) OR toLower(b.id) CONTAINS toLower(a.id))
      AND size(a.id) > 3 AND size(b.id) > 3
      AND ANY(label IN labels(a) WHERE label <> '__Entity__' AND label IN labels(b))
    WITH a, b,
         CASE WHEN size(a.id) >= size(b.id) THEN a ELSE b END AS keeper,
         CASE WHEN size(a.id) >= size(b.id) THEN b ELSE a END AS duplicate
    RETURN keeper.id AS keep, duplicate.id AS merge
    LIMIT 200
    """

    try:
        results = graph.query(FIND_DUPES)
    except Exception as e:
        print(f"  Dedup query failed: {e}")
        return

    if not results:
        print("  No duplicates found")
        return

    print(f"  Found {len(results)} duplicates:")
    merged = 0
    for row in results:
        keep, merge = row["keep"], row["merge"]
        print(f"    '{merge}' -> '{keep}'")
        try:
            graph.query(
                "MATCH (dup) WHERE dup.id = $dup_id DETACH DELETE dup",
                params={"dup_id": merge}
            )
            merged += 1
        except Exception as e:
            print(f"    [ERROR] {e}")

    print(f"  Merged {merged} duplicates")


# Post-Ingestion: Graph Stats
def print_graph_stats(graph):
    """Print summary stats of the built graph."""
    print("\n" + "=" * 60)
    print("GRAPH STATISTICS")
    print("=" * 60)

    try:
        node_count = graph.query("MATCH (n) RETURN count(n) AS c")[0]["c"]
        edge_count = graph.query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
        print(f"Total nodes: {node_count}")
        print(f"Total edges: {edge_count}")

        label_counts = graph.query("""
            MATCH (n) 
            UNWIND labels(n) AS label 
            WHERE label <> '__Entity__'
            RETURN label, count(*) AS cnt 
            ORDER BY cnt DESC
        """)
        print("\nNodes by label:")
        for row in label_counts:
            print(f"  {row['label']}: {row['cnt']}")

        rel_counts = graph.query("""
            MATCH ()-[r]->() 
            RETURN type(r) AS rel, count(*) AS cnt 
            ORDER BY cnt DESC
        """)
        print("\nEdges by type:")
        for row in rel_counts:
            print(f"  {row['rel']}: {row['cnt']}")

        pub_count = graph.query(
            "MATCH (p:Publication) RETURN count(p) AS c"
        )[0]["c"]
        reported_by = graph.query(
            "MATCH ()-[r:REPORTED_BY]->() RETURN count(r) AS c"
        )[0]["c"]
        print(f"\nPublication nodes: {pub_count}")
        print(f"REPORTED_BY edges: {reported_by}")

    except Exception as e:
        print(f"Stats query failed: {e}")


# Full Pipeline
def run_full_pipeline(schema_path, evidence_corpus_path, model="gpt-4o", batch_size=10):
    """
    Complete pipeline:
    1. Load evidence corpus
    2. Chunk with metadata injection
    3. Extract graph with gpt-4o (with schema descriptions in prompt)
    4. Create Publication edges (safety net)
    5. Deduplicate entities
    6. Print stats
    """
    print("=" * 60)
    print("Knowledge Graph Builder v2")
    print("=" * 60)

    df = pd.read_json(evidence_corpus_path, orient='records')
    print(f"Loaded {len(df)} evidence articles")

    chunks = get_chunked_dataset(df)

    builder = KnowledgeGraphBuilder(schema_path, model=model)
    failed = build_graph_with_fallback(chunks, builder, batch_size=batch_size)

    graph = GraphStoreManager.get_neo4j_graph()
    create_publication_edges(graph, df)
    run_deduplication(graph)
    print_graph_stats(graph)

    print("\n" + "=" * 60)
    print(f"Pipeline complete! Failed chunks: {len(failed)}")
    print("=" * 60)

    return failed


# main
if __name__ == "__main__":
    SCHEMA_PATH = './src/scripts/schema_v2.json'
    EVIDENCE_CORPUS_PATH = './data/evidence_corpus.json'
    MODEL = "gpt-4o"
    BATCH_SIZE = 5

    run_full_pipeline(
        schema_path=SCHEMA_PATH,
        evidence_corpus_path=EVIDENCE_CORPUS_PATH,
        model=MODEL,
        batch_size=BATCH_SIZE
    )