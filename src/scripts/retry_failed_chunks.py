"""
Retry Failed Chunks

Reads failed_chunks_v2.json (chunks that hit 429 rate limits during the 
main extraction) and re-processes them with gpt-4o-mini.
 
Since extraction uses MERGE, re-running already-processed chunks is safe 
but wasteful. This script only processes the failures.

"""
 
import time
import json
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from tqdm import tqdm
from src.db.graph_store import GraphStoreManager
from src.configs.config import settings
 
 
def get_nodes_and_edges(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return [n['label'] for n in data['nodes']], [e['label'] for e in data['edges']]
 
 
def build_additional_instructions(schema_path):
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
    lines.append("1. REPORTED_BY is the MOST IMPORTANT edge. Every chunk starts with [Source: X] — ALWAYS create (Entity)-[REPORTED_BY]->(Publication).")
    lines.append("2. NEVER invent relationship types not listed above.")
    lines.append("3. Use CEO_OF for CEO roles, WORKS_FOR for all other employment including athletes.")
    lines.append("4. Use FOUNDED, OWNS, INVESTED_IN distinctly — they are not interchangeable.")
 
    return "\n".join(lines)
 
 
def retry_failed_chunks(
    failed_chunks_file: str,
    evidence_corpus_path: str,
    schema_path: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 10,
    sleep_between: float = 2.0
):
    """
    Re-process chunks that failed during the initial extraction.
    
    Since failed_chunks_v2.json only has metadata (chunk_id, title) not the
    full text, we reconstruct the chunks from the evidence corpus.
    """
    print("=" * 60)
    print("RETRY FAILED CHUNKS")
    print("=" * 60)
 
    # Load failed chunk metadata
    with open(failed_chunks_file, 'r') as f:
        failed = json.load(f)
    
    print(f"Loaded {len(failed)} failed chunks from {failed_chunks_file}")
 
    # Extract specific chunk IDs that failed (format: doc_N_chunk_M)
    failed_chunk_ids = set()
    for item in failed:
        cid = item.get('chunk_id', '')
        if cid:
            failed_chunk_ids.add(cid)
    
    print(f"Unique chunk_ids to retry: {len(failed_chunk_ids)}")
 
    # Load evidence corpus — must preserve same order as original pipeline
    df = pd.read_json(evidence_corpus_path, orient='records')
    df = df.reset_index(drop=True)
    # CRITICAL: use same doc_id scheme as main pipeline
    df['doc_id'] = [f"doc_{i}" for i in range(len(df))]
    
    # Only process articles whose doc_ids appear in failed chunk IDs
    # chunk_id format: "doc_41_chunk_3" -> doc_id is "doc_41"
    failed_doc_ids = set()
    for cid in failed_chunk_ids:
        # Split "doc_41_chunk_3" into "doc_41"
        parts = cid.rsplit("_chunk_", 1)
        if len(parts) == 2:
            failed_doc_ids.add(parts[0])
    
    print(f"Unique source docs to retry: {len(failed_doc_ids)}")
    
    df_failed = df[df['doc_id'].isin(failed_doc_ids)]
    print(f"Matched to {len(df_failed)} articles in evidence corpus")
    
    if len(df_failed) == 0:
        print("ERROR: Could not match failed chunks to corpus articles")
        return
    
    # Re-chunk and keep ONLY the specific failed chunks
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    
    retry_docs = []
    for _, row in df_failed.iterrows():
        raw_text = row.get('body', '')
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue
 
        title = str(row.get('title', '')).strip()
        source = str(row.get('source', '')).strip()
        published_at = str(row.get('published_at', '')).strip()
        if 'T' in published_at:
            published_at = published_at.split('T')[0]
 
        meta_parts = []
        if source: meta_parts.append(f"Source: {source}")
        if title: meta_parts.append(f"Title: {title}")
        if published_at: meta_parts.append(f"Published: {published_at}")
        prefix = f"[{' | '.join(meta_parts)}]\n\n" if meta_parts else ""
 
        chunks = text_splitter.split_text(raw_text)
        doc_id = row['doc_id']
        
        for i, chunk_text in enumerate(chunks):
            # Reconstruct the exact chunk_id from original pipeline
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # ONLY keep chunks that actually failed
            if chunk_id not in failed_chunk_ids:
                continue
            
            retry_docs.append(Document(
                page_content=prefix + chunk_text,
                metadata={
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                    "title": title,
                    "source": source,
                    "published_at": published_at
                }
            ))
    
    print(f"Reconstructed {len(retry_docs)} chunks to retry (only failed ones)")
 
    # Set up the transformer with the new model
    print(f"\nUsing model: {model}")
    llm = ChatOpenAI(temperature=0, model=model, api_key=settings.OPENAI_API_KEY)
    
    nodes, edges = get_nodes_and_edges(schema_path)
    instructions = build_additional_instructions(schema_path)
    
    transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=nodes,
        allowed_relationships=edges,
        strict_mode=True,
        node_properties=False,
        relationship_properties=False,
        additional_instructions=instructions,
    )
 
    graph_db = GraphStoreManager.get_neo4j_graph()
 
    # Process in batches
    still_failed = []
    total_triples = 0
    
    for i in tqdm(range(0, len(retry_docs), batch_size), desc="Retrying"):
        batch = retry_docs[i:i + batch_size]
        
        try:
            graph_docs = transformer.convert_to_graph_documents(batch)
            batch_triples = sum(len(gd.relationships) for gd in graph_docs)
            total_triples += batch_triples
            
            graph_db.add_graph_documents(
                graph_docs,
                baseEntityLabel=True,
                include_source=True
            )
            time.sleep(sleep_between)
            
        except Exception as e:
            print(f"\n[ERROR] Batch {i}-{i+len(batch)}: {str(e)[:150]}")
            for doc in batch:
                still_failed.append({
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "title": doc.metadata.get("title"),
                    "error": str(e)[:200]
                })
            continue
    
    print(f"\nExtracted ~{total_triples} new triples")
    
    if still_failed:
        print(f"\n[WARN] {len(still_failed)} chunks still failed")
        with open("still_failed_chunks.json", 'w') as f:
            json.dump(still_failed, f, indent=2)
        print("Saved to still_failed_chunks.json")
    else:
        print("\n[SUCCESS] All retry chunks processed!")
    
    return still_failed
 
#  Main
if __name__ == "__main__":
    FAILED_CHUNKS_FILE = "./data/failed_chunks_v2.json"
    EVIDENCE_CORPUS_PATH = "./data/evidence_corpus.json"
    SCHEMA_PATH = "./src/scripts/schema_v2.json"
    
    retry_failed_chunks(
        failed_chunks_file=FAILED_CHUNKS_FILE,
        evidence_corpus_path=EVIDENCE_CORPUS_PATH,
        schema_path=SCHEMA_PATH,
        model="gpt-4o",
        batch_size=5,
        sleep_between=5
    )
