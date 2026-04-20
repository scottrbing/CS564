from langchain_chroma import Chroma
from src.configs.config import embeddings, settings

class VectorStoreManager:

    _instance = None

    @classmethod
    def get_chroma_store(cls):
        if cls._instance is None:
            print("Initializing vector store...")
            cls._instance = Chroma(
                collection_name = settings.CHROMA_COLLECTION_NAME,
                embedding_function = embeddings,
                persist_directory = settings.CHROMA_PERSIST_DIRECTORY,
            )
        return cls._instance
    
chroma_db = VectorStoreManager.get_chroma_store()

