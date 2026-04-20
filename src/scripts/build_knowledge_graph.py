from src.scripts.common import *
from src.scripts.graph_transformer import KnowledgeGraphBuilder, build_graph_with_fallback, get_chunked_dataset, get_nodes_and_edges
from src.db.graph_store import GraphStoreManager

print("Careful running this script as it will attempt to build a knowledge graph for every document in the corpus, which can be time consuming and costly.")
schema_path = './data/schema/1_clean.json'
builder = KnowledgeGraphBuilder(schema_path)

df, _ = get_corpus_and_queries()
chks = get_chunked_dataset(df)
failed_chks = build_graph_with_fallback(chks, builder)
print(failed_chks)