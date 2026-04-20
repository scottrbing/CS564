import pandas as pd
from src.configs.config import settings
import re
import os

def get_corpus_and_queries():
    df_corpus = pd.read_json(settings.DOCUMENT_COLLECTION_PATH)
    df_queries = pd.read_json(settings.QUERIES_COLLECTION_PATH)

    def clean_article_text(raw_text: str) -> str:
        if not isinstance(raw_text, str):
            return ""    
        cleaned = raw_text
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        return cleaned.strip()

    df_corpus['body'] = df_corpus['body'].apply(clean_article_text)
    df_corpus['published_at'] = df_corpus['published_at'].astype('str')
    return df_corpus, df_queries

def upsert_test_set(queries_df, test_set_path, num_samples_per_category=10):
    """
    Safely samples new queries from the master dataset and appends them to the tracking JSON,
    No duplicate queries are ever added.
    """
    
    if os.path.exists(test_set_path):
        test_df = pd.read_json(test_set_path, orient="records")
        existing_queries = set(test_df['query'].tolist())
        print(f"Loaded existing test set with {len(test_df)} queries.")
    else:
        test_df = pd.DataFrame()
        existing_queries = set()
        print("Test set doesn't exist. Creating a new one")

    available_queries_df = queries_df[~queries_df['query'].isin(existing_queries)].copy()
    
    if available_queries_df.empty:
        print("No new queries available! Completely tested all available queries.")
        return test_df
        
    new_samples = []
    for _, group in available_queries_df.groupby('question_type'):
        sample_size = min(num_samples_per_category, len(group))
        if sample_size > 0:
            sampled_group = group.sample(n=sample_size) 
            new_samples.append(sampled_group)
            
    if not new_samples:
        print("Could not find any new samples")
        return test_df
        
    new_samples_df = pd.concat(new_samples, ignore_index=True)
    new_samples_df['vector_rag_status'] = 'PENDING'
    new_samples_df['graph_rag_status'] = 'PENDING'
    new_samples_df['hybrid_rag_status'] = 'PENDING'
    new_samples_df['vector_rag_answer'] = None
    new_samples_df['graph_rag_answer'] = None
    new_samples_df['hybrid_rag_answer'] = None
    new_samples_df['vector_rag_latency'] = None
    new_samples_df['graph_rag_latency'] = None
    new_samples_df['hybrid_rag_latency'] = None

    updated_test_df = pd.concat([test_df, new_samples_df], ignore_index=True)
    updated_test_df.to_json(test_set_path, orient="records", indent=4)
    
    print(f"Successfully added {len(new_samples_df)} new queries.")
    print(f"Test set now has {len(updated_test_df)}")
    
    return updated_test_df