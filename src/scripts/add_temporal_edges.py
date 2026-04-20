"""
Add Temporal Layer to Graph
=============================
Creates Date nodes and connects Documents to them via PUBLISHED_ON edges.
Also adds temporal ordering: each Date has month, year, quarter properties.

This enables temporal queries like:
  - "Did TechCrunch report on X before Y?"
  - "What was reported in November 2023?"
  - "Articles between date A and date B"

Why Date nodes instead of properties?
  Graph traversal handles "before/after" relationships natively:
    MATCH (d1:Document)-[:PUBLISHED_ON]->(date1:Date),
          (d2:Document)-[:PUBLISHED_ON]->(date2:Date)
    WHERE date1.iso_date < date2.iso_date
  
  This makes temporal reasoning a graph operation, not a string comparison.
"""

import pandas as pd
from datetime import datetime
from src.db.graph_store import GraphStoreManager


def create_date_nodes(graph, evidence_corpus_path):
    """
    For each unique publication date in the corpus:
    1. Create a Date node with iso_date, month, year, quarter
    2. Connect each Document to its Date via PUBLISHED_ON
    3. Create temporal ordering edges between consecutive Dates
    """
    print("ADDING TEMPORAL LAYER TO GRAPH")
    df = pd.read_json(evidence_corpus_path, orient='records')
    
    # Collect unique dates
    unique_dates = set()
    for _, row in df.iterrows():
        pub_date_raw = row.get('published_at')
        if pub_date_raw is None or pd.isna(pub_date_raw):
            continue
        # Handle pandas Timestamp objects AND string inputs
        if hasattr(pub_date_raw, 'strftime'):
            pub_date = pub_date_raw.strftime('%Y-%m-%d')
        else:
            pub_date = str(pub_date_raw).strip()
            if 'T' in pub_date:
                pub_date = pub_date.split('T')[0]
            elif ' ' in pub_date:
                pub_date = pub_date.split(' ')[0]
        if pub_date and len(pub_date) == 10:
            unique_dates.add(pub_date)
    
    print(f"Found {len(unique_dates)} unique dates")
    
    # Create Date nodes
    date_nodes_created = 0
    for date_str in sorted(unique_dates):
        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            quarter = (dt.month - 1) // 3 + 1
            
            graph.query(
                """
                MERGE (d:Date:__Entity__ {id: $iso_date})
                SET d.iso_date = date($iso_date),
                    d.year = $year,
                    d.month = $month,
                    d.day = $day,
                    d.quarter = $quarter,
                    d.year_month = $year_month
                """,
                params={
                    "iso_date": date_str,
                    "year": dt.year,
                    "month": dt.month,
                    "day": dt.day,
                    "quarter": quarter,
                    "year_month": f"{dt.year}-{dt.month:02d}"
                }
            )
            date_nodes_created += 1
        except Exception as e:
            print(f"  Failed to create Date '{date_str}': {e}")
    
    print(f"Created {date_nodes_created} Date nodes")
    
    # Link Documents to their Dates via PUBLISHED_ON
    print("\nLinking Documents to Dates...")
    linked = 0
    for _, row in df.iterrows():
        title = str(row.get('title', '')).strip()
        pub_date_raw = row.get('published_at')
        
        if pub_date_raw is None or pd.isna(pub_date_raw):
            continue
        if hasattr(pub_date_raw, 'strftime'):
            pub_date = pub_date_raw.strftime('%Y-%m-%d')
        else:
            pub_date = str(pub_date_raw).strip()
            if 'T' in pub_date:
                pub_date = pub_date.split('T')[0]
            elif ' ' in pub_date:
                pub_date = pub_date.split(' ')[0]
        
        if not title or not pub_date or len(pub_date) != 10:
            continue
        
        try:
            result = graph.query(
                """
                MATCH (doc:Document)
                WHERE doc.text CONTAINS $title_fragment
                MATCH (date:Date {id: $date})
                MERGE (doc)-[:PUBLISHED_ON]->(date)
                RETURN count(doc) AS cnt
                """,
                params={"title_fragment": title[:50], "date": pub_date}
            )
            if result and result[0]["cnt"] > 0:
                linked += result[0]["cnt"]
        except Exception as e:
            pass
    
    print(f"Created {linked} PUBLISHED_ON edges")


def create_temporal_ordering(graph):
    """
    Create NEXT_DAY edges between consecutive Date nodes.
    This lets queries traverse time: (Date)-[:NEXT_DAY]->(Date)-[:NEXT_DAY]->(Date)
    """
    print("\nCreating temporal ordering edges...")
    
    graph.query(
        """
        MATCH (d1:Date), (d2:Date)
        WHERE d1.iso_date < d2.iso_date
          AND NOT EXISTS {
            MATCH (d3:Date) 
            WHERE d3.iso_date > d1.iso_date AND d3.iso_date < d2.iso_date
          }
        MERGE (d1)-[:NEXT_DAY]->(d2)
        """
    )
    
    result = graph.query("MATCH ()-[r:NEXT_DAY]->() RETURN count(r) AS cnt")
    print(f"Created {result[0]['cnt']} NEXT_DAY edges")


def verify_temporal_layer(graph):
    """Print stats and verify the temporal layer works."""
    print("\n" + "=" * 60)
    print("TEMPORAL LAYER VERIFICATION")
    print("=" * 60)
    
    # Date coverage
    stats = graph.query(
        """
        MATCH (d:Date)
        RETURN min(d.iso_date) AS earliest, 
               max(d.iso_date) AS latest, 
               count(d) AS total_dates
        """
    )[0]
    print(f"Date range: {stats['earliest']} to {stats['latest']}")
    print(f"Total Date nodes: {stats['total_dates']}")
    
    # Coverage
    coverage = graph.query(
        """
        MATCH (doc:Document)
        OPTIONAL MATCH (doc)-[:PUBLISHED_ON]->(d:Date)
        RETURN count(doc) AS total_docs,
               count(d) AS docs_with_date
        """
    )[0]
    print(f"Documents with dates: {coverage['docs_with_date']}/{coverage['total_docs']}")
    
    # Example query: articles by month
    by_month = graph.query(
        """
        MATCH (d:Date)<-[:PUBLISHED_ON]-(doc:Document)
        RETURN d.year_month AS month, count(doc) AS articles
        ORDER BY month
        """
    )
    print("\nArticles per month:")
    for row in by_month:
        print(f"  {row['month']}: {row['articles']}")
    
    # Test a temporal query
    print("\nExample temporal query:")
    print("  'Articles published in November 2023':")
    nov_2023 = graph.query(
        """
        MATCH (date:Date)<-[:PUBLISHED_ON]-(doc:Document)-[:REPORTED_BY]->(pub:Publication)
        WHERE date.year = 2023 AND date.month = 11
        RETURN pub.id AS publication, count(doc) AS articles
        ORDER BY articles DESC
        LIMIT 5
        """
    )
    for row in nov_2023:
        print(f"    {row['publication']}: {row['articles']} articles")


if __name__ == "__main__":
    graph = GraphStoreManager.get_neo4j_graph()
    
    EVIDENCE_PATH = "./data/evidence_corpus.json"
    
    create_date_nodes(graph, EVIDENCE_PATH)
    create_temporal_ordering(graph)
    verify_temporal_layer(graph)
    
    print("\n" + "=" * 60)
    print("Temporal layer complete!")
    print("=" * 60)
    print("\nNew edge types to use in retrieval:")
    print("  - (Document)-[:PUBLISHED_ON]->(Date)")
    print("  - (Date)-[:NEXT_DAY]->(Date)  [for ordering]")
    print("\nDate node properties: iso_date, year, month, day, quarter, year_month")