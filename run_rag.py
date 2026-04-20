
import pandas as pd
from src.bots.graph_retrieval_v3 import GraphRAGRetriever
from src.configs.config import settings
from src.bots.models import ask_vector_rag_direct
import time
from src.db.graph_store import graph_db
from src.db.vector_store import chroma_db
from src.bots.hybrid_retrieval import HybridGraphVectorRetriever

graph_retriever = GraphRAGRetriever(graph_db, settings.OPENAI_API_KEY)
hybrid_retriever = HybridGraphVectorRetriever(graph_db, chroma_db, settings.OPENAI_API_KEY)


path_to_test_set = "./results/test_set.json"
df_test = pd.read_json(path_to_test_set, orient="records")

for index, item in df_test.iterrows():
    query = item['query']
    
    if item['vector_rag_status'] == 'PENDING':
        print(f"\n[{index}] Vector RAG processing...")
        try:
            start_time = time.perf_counter()
            answer = ask_vector_rag_direct(query)
            
            latency = time.perf_counter() - start_time
            
            df_test.at[index, 'vector_rag_answer'] = answer
            df_test.at[index, 'vector_rag_latency'] = round(latency, 2)
            df_test.at[index, 'vector_rag_status'] = 'SUCCESS'
            
        except Exception as e:
            print(f"Vector RAG Error: {str(e)}")
            df_test.at[index, 'vector_rag_status'] = 'FAILED'

    if item['graph_rag_status'] == 'PENDING':
        print(f"[{index}] Graph RAG processing...")
        try:
            start_time = time.perf_counter()
            
            answer = graph_retriever.answer(query, verbose=True)
            
            latency = time.perf_counter() - start_time
            print(f'Graph RAG answer: {answer}')
        
            df_test.at[index, 'graph_rag_answer'] = answer
            df_test.at[index, 'graph_rag_latency'] = round(latency, 2)
            df_test.at[index, 'graph_rag_status'] = 'SUCCESS'
        
        except Exception as e:
            print(f"Graph RAG Error: {str(e)}")
            df_test.at[index, 'graph_rag_status'] = 'FAILED'
    
    if item['hybrid_rag_status'] == 'PENDING':
        print(f"[{index}] Hybrid RAG processing...")
        try:
            start_time = time.perf_counter()
            
            answer = hybrid_retriever.answer(query, verbose=True)
            
            latency = time.perf_counter() - start_time
            print(f'Hybrid RAG answer: {answer}')
        
            df_test.at[index, 'hybrid_rag_answer'] = answer
            df_test.at[index, 'hybrid_rag_latency'] = round(latency, 2)
            df_test.at[index, 'hybrid_rag_status'] = 'SUCCESS'
        
        except Exception as e:
            print(f"Hybrid RAG Error: {str(e)}")
            df_test.at[index, 'hybrid_rag_status'] = 'FAILED'
    
    df_test.to_json(path_to_test_set, orient="records", indent=2)
