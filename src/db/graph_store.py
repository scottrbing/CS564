import os
from langchain_community.graphs import Neo4jGraph
from src.configs.config import settings # Assuming you have this from your vector setup

class GraphStoreManager:
    _instance = None

    @classmethod
    def get_neo4j_graph(cls):
        """Initializes and returns the Singleton Neo4jGraph connection."""
        if cls._instance is None:
            print("Initializing Neo4j Graph Database Connection...")
            try:
                print(f'Database name: {settings.NEO4J_DATABASE}')
                cls._instance = Neo4jGraph(url=settings.NEO4J_URI,
                                        username=settings.NEO4J_USERNAME,
                                        password=settings.NEO4J_PASSWORD,
                                        database=settings.NEO4J_DATABASE
                                        )
                print("Connection successful!")
            except Exception as e:
                print(f"Failed to connect to Neo4j. Error: {e}")
                raise e
                
        return cls._instance

    @classmethod
    def verify_connection(cls):
        """Runs a lightweight Cypher query to ensure the DB is responsive."""
        graph = cls.get_neo4j_graph()
        try:
            result = graph.query("MATCH (n) RETURN count(n) AS node_count")
            count = result[0]['node_count']
            print(f"Connection Verified! Current Node Count in DB: {count}")
            return True
        except Exception as e:
            print(f"Connection failed during verification: {e}")
            return False

graph_db = GraphStoreManager.get_neo4j_graph()