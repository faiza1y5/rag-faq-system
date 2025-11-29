"""
Unit tests for RAG system
"""

import pytest
from app.rag.query_engine import get_query_engine
from app.rag.vector_store import get_vector_store
from app.models.schemas import QueryRequest


def test_vector_store_initialized():
    """Test that vector store is properly initialized"""
    vector_store = get_vector_store()
    count = vector_store.get_count()
    assert count > 0, "Vector store should contain documents"


def test_query_engine_initialization():
    """Test that query engine initializes properly"""
    query_engine = get_query_engine()
    assert query_engine is not None
    assert query_engine.embedding_model is not None
    assert query_engine.vector_store is not None


def test_simple_query():
    """Test a simple query"""
    query_engine = get_query_engine()
    response = query_engine.query("What are your office hours?")
    
    assert response.answer is not None
    assert len(response.answer) > 0
    assert response.confidence >= 0.0
    assert response.confidence <= 1.0


def test_insurance_query():
    """Test insurance-related query"""
    query_engine = get_query_engine()
    response = query_engine.query("Do you accept Blue Cross insurance?")
    
    assert response.answer is not None
    assert len(response.sources) > 0
    assert any("insurance" in source.content.lower() for source in response.sources)


def test_parking_query():
    """Test parking-related query"""
    query_engine = get_query_engine()
    response = query_engine.query("Where can I park?")
    
    assert response.answer is not None
    assert len(response.sources) > 0


def test_irrelevant_query():
    """Test handling of irrelevant query"""
    query_engine = get_query_engine()
    response = query_engine.query("What is the weather like today?")
    
    assert response.answer is not None
    # Should have low confidence or explicit "don't know" response
    assert response.confidence < 0.7 or "don't" in response.answer.lower()


def test_empty_query():
    """Test handling of empty query"""
    query_engine = get_query_engine()
    response = query_engine.query("")
    
    assert response.answer is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
