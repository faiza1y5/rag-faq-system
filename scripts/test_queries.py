"""
Test script to verify RAG system with sample queries
"""

import sys
from pathlib import Path
import asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.rag.query_engine import get_query_engine
from app.utils.logger import get_logger

logger = get_logger(__name__)


SAMPLE_QUERIES = [
    "What insurance do you accept?",
    "Where can I park?",
    "What are your office hours?",
    "What should I bring to my first appointment?",
    "Do you accept walk-ins?",
    "What is your cancellation policy?",
    "How much does a general consultation cost?",
    "Do you offer telehealth appointments?",
    "What COVID-19 protocols do you have?",
    "How do I get my test results?",
]


def test_queries():
    """Test the RAG system with sample queries"""
    logger.info("=" * 80)
    logger.info("Testing RAG FAQ System")
    logger.info("=" * 80)
    
    query_engine = get_query_engine()
    
    for i, question in enumerate(SAMPLE_QUERIES, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Query {i}/{len(SAMPLE_QUERIES)}")
        logger.info(f"{'='*80}")
        logger.info(f"Question: {question}")
        logger.info("-" * 80)
        
        try:
            response = query_engine.query(question)
            
            logger.info(f"Answer: {response.answer}")
            logger.info(f"\nConfidence: {response.confidence:.2%}")
            logger.info(f"Sources Used: {len(response.sources)}")
            
            if response.sources:
                logger.info("\nSource Documents:")
                for j, source in enumerate(response.sources, 1):
                    logger.info(f"  {j}. Similarity: {source.similarity_score:.3f}")
                    logger.info(f"     Category: {source.metadata.get('category', 'N/A')}")
                    logger.info(f"     Content: {source.content[:100]}...")
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
        
        logger.info("")
    
    logger.info("=" * 80)
    logger.info("Testing Complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        test_queries()
    except Exception as e:
        logger.error(f"Testing failed: {e}", exc_info=True)
        sys.exit(1)