"""
Enrich Graph with Node Properties
===================================
Adds structured properties to nodes so Cypher queries can filter on them:

- Document nodes: published_at, category, source
- Publication nodes: article_count
- Event nodes: date (parsed from context — best effort)

Why this matters:
  Temporal queries like "Did X happen before Y?" become answerable in Cypher:
    MATCH (a:Event)-[:OCCURRED_ON]->(d1:Date), (b:Event)-[:OCCURRED_ON]->(d2:Date)
    WHERE d1.iso_date < d2.iso_date
    RETURN ...
"""

import json
import pandas as pd
from src.db.graph_store import GraphStoreManager


def enrich_document_nodes(graph, evidence_corpus_path):
    """
    Add published_at, category, and source properties to Document nodes.
    Matches Documents to evidence articles by title content.
    """
    print("=" * 60)
    print("ENRICHING DOCUMENT NODES WITH PROPERTIES")
    print("=" * 60)
    
    df = pd.read_json(evidence_corpus_path, orient='records')
    
    enriched = 0
    for _, row in df.iterrows():
        title = str(row.get('title', '')).strip()
        source = str(row.get('source', '')).strip()
        category = str(row.get('category', '')).strip()
        
        # Handle pandas Timestamp AND string inputs
        pub_date_raw = row.get('published_at')
        if pub_date_raw is None or pd.isna(pub_date_raw):
            published_at = ''
        elif hasattr(pub_date_raw, 'strftime'):
            published_at = pub_date_raw.strftime('%Y-%m-%d')
        else:
            published_at = str(pub_date_raw).strip()
            if 'T' in published_at:
                published_at = published_at.split('T')[0]
            elif ' ' in published_at:
                published_at = published_at.split(' ')[0]
        
        if not title:
            continue
        
        # Update Document nodes whose text contains this title
        try:
            result = graph.query(
                """
                MATCH (d:Document)
                WHERE d.text CONTAINS $title_fragment
                SET d.source = $source,
                    d.published_at = date($published_at),
                    d.category = $category,
                    d.title = $title
                RETURN count(d) AS cnt
                """,
                params={
                    "title_fragment": title[:50],
                    "source": source,
                    "published_at": published_at if published_at else "2023-01-01",
                    "category": category,
                    "title": title
                }
            )
            if result and result[0]["cnt"] > 0:
                enriched += result[0]["cnt"]
        except Exception as e:
            # If date parsing fails, try without it
            try:
                graph.query(
                    """
                    MATCH (d:Document)
                    WHERE d.text CONTAINS $title_fragment
                    SET d.source = $source,
                        d.category = $category,
                        d.title = $title
                    """,
                    params={
                        "title_fragment": title[:50],
                        "source": source,
                        "category": category,
                        "title": title
                    }
                )
            except Exception as e2:
                print(f"  Failed for '{title[:50]}': {e2}")
    
    print(f"Enriched {enriched} Document nodes with properties")


def enrich_publication_nodes(graph):
    """Add article_count to Publication nodes."""
    print("\n[ENRICH] Adding article_count to Publications...")
    
    graph.query(
        """
        MATCH (p:Publication)
        OPTIONAL MATCH (d:Document)-[:REPORTED_BY]->(p)
        WITH p, count(d) AS article_count
        SET p.article_count = article_count
        """
    )
    
    # Verify
    result = graph.query(
        """
        MATCH (p:Publication) 
        WHERE p.article_count IS NOT NULL
        RETURN p.id AS pub, p.article_count AS count
        ORDER BY p.article_count DESC
        LIMIT 10
        """
    )
    
    print("Top publications by article count:")
    for row in result:
        print(f"  {row['pub']}: {row['count']}")


if __name__ == "__main__":
    graph = GraphStoreManager.get_neo4j_graph()
    
    EVIDENCE_PATH = "./data/evidence_corpus.json"
    
    enrich_document_nodes(graph, EVIDENCE_PATH)
    enrich_publication_nodes(graph)
    
    print("\nDone! Document and Publication nodes now have structured properties.")