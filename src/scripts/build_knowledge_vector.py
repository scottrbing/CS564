import pandas as pd
from src.configs.config import settings
import re

from langchain_community.document_loaders import DataFrameLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.db.vector_store import chroma_db

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


# Split the documents into chunks. 1000 characters with an overlap of 150 characters.
text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

docs = DataFrameLoader(data_frame=df_corpus, page_content_column= 'body').load()
chunked_docs = text_splitter.split_documents(docs)
batch_size = 5000
for i in range(0, len(chunked_docs), batch_size):
    batch = chunked_docs[i : i + batch_size]
    print(f"Inserting batch {i} to {i + len(batch)}...")
    chroma_db.add_documents(batch)