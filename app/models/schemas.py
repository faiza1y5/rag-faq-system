from pydantic import BaseModel, Field
from typing import List, Optional


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User's question")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")


class SourceDocument(BaseModel):
    content: str = Field(..., description="Document content")
    metadata: dict = Field(default_factory=dict, description="Document metadata")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    question: str = Field(..., description="Original question")


class HealthResponse(BaseModel):
    status: str
    version: str
    vector_db_status: str
