"""
Script to initialize vector database with clinic FAQ data
Run this once to set up your RAG system
"""

import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_faq_data(faq_path: str = "./data/clinic_faq.json") -> dict:
    """Load FAQ data from JSON file"""
    with open(faq_path, 'r') as f:
        return json.load(f)


def flatten_faq_data(faq_data: dict, parent_key: str = '') -> list:
    """Flatten nested FAQ data into chunks"""
    chunks = []
    
    for key, value in faq_data.items():
        new_key = f"{parent_key}.{key}" if parent_key else key
        
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            chunks.extend(flatten_faq_data(value, new_key))
        elif isinstance(value, list):
            # Convert lists to text
            text = f"{new_key.replace('_', ' ').title()}:\n" + "\n".join(f"- {item}" for item in value)
            chunks.append({
                "text": text,
                "metadata": {"category": parent_key.split('.')[0], "subcategory": new_key}
            })
        else:
            # Simple key-value pair
            text = f"{new_key.replace('_', ' ').title()}: {value}"
            chunks.append({
                "text": text,
                "metadata": {"category": parent_key.split('.')[0], "subcategory": new_key}
            })
    
    return chunks


def setup_vector_database():
    """Main setup function"""
    logger.info("=" * 60)
    logger.info("Starting Vector Database Setup")
    logger.info("=" * 60)
    
    # Load FAQ data
    logger.info("Loading FAQ data...")
    faq_data = load_faq_data()
    logger.info(f"Loaded FAQ data with {len(faq_data)} top-level categories")
    
    # Flatten data into chunks
    logger.info("Processing FAQ data into chunks...")
    chunks = flatten_faq_data(faq_data)
    logger.info(f"Created {len(chunks)} document chunks")
    
    # Initialize embedding model
    logger.info("Initializing embedding model...")
    embedding_model = get_embedding_model()
    
    # Generate embeddings
    logger.info("Generating embeddings (this may take a minute)...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embedding_model.encode(texts)
    logger.info(f"Generated {len(embeddings)} embeddings")
    
    # Initialize vector store
    logger.info("Initializing vector store...")
    vector_store = get_vector_store()
    
    # Reset collection (clear old data)
    logger.info("Clearing old data...")
    vector_store.reset()
    
    # Add documents to vector store
    logger.info("Adding documents to vector store...")
    vector_store.add_documents(
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[chunk["metadata"] for chunk in chunks],
        ids=[f"doc_{i}" for i in range(len(chunks))]
    )
    
    # Verify
    doc_count = vector_store.get_count()
    logger.info(f"Vector store now contains {doc_count} documents")
    
    logger.info("=" * 60)
    logger.info("Setup Complete!")
    logger.info("=" * 60)
    logger.info("You can now start the API server with: python main.py")


if __name__ == "__main__":
    try:
        setup_vector_database()
    except Exception as e:
        logger.error(f"Setup failed: {e}", exc_info=True)
        sys.exit(1)