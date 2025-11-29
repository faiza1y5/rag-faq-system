import chromadb
from chromadb.config import Settings as ChromaSettings
from typing import List, Dict, Optional
import json
from pathlib import Path
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    def __init__(self):
        self.db_path = Path(settings.VECTOR_DB_PATH)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing ChromaDB at: {self.db_path}")
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        
        self.collection_name = settings.COLLECTION_NAME
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(name=self.collection_name)
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        return collection
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the vector store"""
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        if metadatas is None:
            metadatas = [{} for _ in documents]
        
        try:
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = None,
        where: Optional[Dict] = None
    ) -> Dict:
        """Query the vector store"""
        if top_k is None:
            top_k = settings.TOP_K_RESULTS
        
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where
            )
            logger.debug(f"Retrieved {len(results['documents'][0])} results")
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            raise
    
    def get_count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()
    
    def reset(self):
        """Reset the collection"""
        self.client.delete_collection(name=self.collection_name)
        self.collection = self._get_or_create_collection()
        logger.info("Collection reset successfully")


# Singleton instance
vector_store = None


def get_vector_store() -> VectorStore:
    """Get or create vector store instance"""
    global vector_store
    if vector_store is None:
        vector_store = VectorStore()
    return vector_store