from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    def __init__(self):
        self.model_name = settings.EMBEDDING_MODEL
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts"""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """Generate embedding for a single text"""
        return self.encode([text])[0]


# Singleton instance
embedding_model = None


def get_embedding_model() -> EmbeddingModel:
    """Get or create embedding model instance"""
    global embedding_model
    if embedding_model is None:
        embedding_model = EmbeddingModel()
    return embedding_model