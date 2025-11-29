from fastapi import APIRouter, HTTPException, status
from app.models.schemas import QueryRequest, QueryResponse, HealthResponse
from app.rag.query_engine import get_query_engine
from app.rag.vector_store import get_vector_store
from app.utils.logger import get_logger
from config.settings import settings

logger = get_logger(__name__)
router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask_question(request: QueryRequest):
    """
    Ask a question about the clinic
    
    Example request:
    ```json
    {
        "question": "What insurance do you accept?"
    }
    ```
    """
    try:
        logger.info(f"Received question: {request.question}")
        query_engine = get_query_engine()
        response = query_engine.query(request.question)
        return response
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing your question. Please try again."
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        vector_store = get_vector_store()
        doc_count = vector_store.get_count()
        
        return HealthResponse(
            status="healthy",
            version=settings.APP_VERSION,
            vector_db_status=f"operational ({doc_count} documents)"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version=settings.APP_VERSION,
            vector_db_status="error"
        )


@router.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"Welcome to {settings.CLINIC_NAME} FAQ API",
        "version": settings.APP_VERSION,
        "endpoints": {
            "ask": "/api/ask",
            "health": "/api/health",
            "docs": "/docs"
        }
    }
