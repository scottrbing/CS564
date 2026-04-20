"""
Merge Duplicate Publications
=============================
Merges duplicate Publication nodes that resulted from case/name variations
during extraction.

Run in Neo4j Desktop or via graph.query() in Python.
"""

# Known duplicates from the graph stats, retrieved manually
# Format: (keep_this, merge_these)
MERGE_MAP = [
    ("Espn", ["Espn.Com"]),
    ("CBSSports.com", ["Cbssports.Com"]),
    
    ("Globes English | Israel Business Arena", ["Globes English", "Globes"]),
    ("Live Science: The Most Interesting Articles", ["Live Science"]),
    ("Scitechdaily | Science Space And Technology News 2017", ["Scitechdaily"]),
    ("Business Today | Latest Stock Market And Economy News India", ["Business Today"]),
    
    ("Fox News", ["FOX News - Health", "FOX News - Lifestyle", "FOX News - Entertainment"]),
    ("The Sporting News", ["Sporting News"]),
    
    ("The Roar | Sports Writers Blog", ["The Roar"]),
]


def merge_publications(graph):
    """
    For each (keeper, duplicates) pair:
    1. Re-point REPORTED_BY edges from duplicate -> keeper
    2. Delete the duplicate node
    """
    print("MERGING DUPLICATE PUBLICATIONS")
    
    total_merged = 0
    
    for keeper_id, duplicate_ids in MERGE_MAP:
        for dup_id in duplicate_ids:
            if not dup_id:
                continue
            
            # Check if both exist
            check = graph.query(
                """
                MATCH (keeper:Publication {id: $keeper_id})
                MATCH (dup:Publication {id: $dup_id})
                RETURN keeper.id AS k, dup.id AS d
                """,
                params={"keeper_id": keeper_id, "dup_id": dup_id}
            )
            
            if not check:
                print(f"  SKIP: '{dup_id}' or '{keeper_id}' not found")
                continue
            
            # Count edges to move
            edge_count = graph.query(
                """
                MATCH (n)-[r:REPORTED_BY]->(dup:Publication {id: $dup_id})
                RETURN count(r) AS cnt
                """,
                params={"dup_id": dup_id}
            )[0]["cnt"]
            
            # Re-point edges
            graph.query(
                """
                MATCH (n)-[r:REPORTED_BY]->(dup:Publication {id: $dup_id})
                MATCH (keeper:Publication {id: $keeper_id})
                MERGE (n)-[:REPORTED_BY]->(keeper)
                DELETE r
                """,
                params={"dup_id": dup_id, "keeper_id": keeper_id}
            )
            
            # Delete the duplicate
            graph.query(
                """
                MATCH (dup:Publication {id: $dup_id})
                DETACH DELETE dup
                """,
                params={"dup_id": dup_id}
            )
            
            print(f"  MERGED: '{dup_id}' -> '{keeper_id}' ({edge_count} edges moved)")
            total_merged += 1
    
    print(f"\nTotal duplicates merged: {total_merged}")


def find_potential_duplicates(graph):
    """Find Publication nodes that look similar — run before manual merge."""
    print("POTENTIAL DUPLICATE PUBLICATIONS (for manual review)")
    # Find publications where one name contains another
    results = graph.query(
        """
        MATCH (a:Publication), (b:Publication)
        WHERE a.id <> b.id
          AND id(a) < id(b)
          AND (toLower(a.id) CONTAINS toLower(b.id) 
               OR toLower(b.id) CONTAINS toLower(a.id))
        RETURN a.id AS pub_a, b.id AS pub_b
        ORDER BY size(a.id) + size(b.id)
        """
    )
    
    for row in results:
        print(f"  '{row['pub_a']}' <-> '{row['pub_b']}'")
    
    return results

#  Main
if __name__ == "__main__":
    from src.db.graph_store import GraphStoreManager
    
    graph = GraphStoreManager.get_neo4j_graph()
    
    # First, find all potential duplicates
    find_potential_duplicates(graph)
    
    # Then merge the known ones
    print("\nProceeding with MERGE_MAP...")
    merge_publications(graph)
    
    # Final check
    print("\n" + "=" * 60)
    print("FINAL PUBLICATION COUNT")
    print("=" * 60)
    result = graph.query("MATCH (p:Publication) RETURN count(p) AS cnt")
    print(f"Publications remaining: {result[0]['cnt']}")
