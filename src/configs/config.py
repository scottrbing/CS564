from langchain_openai import OpenAIEmbeddings
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppSettings(BaseSettings):

    PROJECT_NAME: str = "Vector-RAG vs Graph-RAG"
    ENVIRONMENT: str = "development"
    
    # API Keys (No default value means it will throw an error if missing)
    OPENAI_API_KEY: str

    # Database Settings
    CHROMA_PERSIST_DIRECTORY: str = "./db/chroma_langchain_db"
    CHROMA_COLLECTION_NAME: str = "docs_knowledge"

    # LLM & Vector Settings
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    CHAT_MODEL: str = "gpt-4o-mini"

    # Dataset paths
    DOCUMENT_COLLECTION_PATH: str
    QUERIES_COLLECTION_PATH: str


    # This tells Pydantic to look for a .env file in the root directory
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = AppSettings()

embeddings = OpenAIEmbeddings(api_key = settings.OPENAI_API_KEY ,model = "text-embedding-3-small")