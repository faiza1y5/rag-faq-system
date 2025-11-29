from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # Application
    APP_NAME: str = "Medical Clinic RAG FAQ"
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = "development"
    LOG_LEVEL: str = "INFO"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # LLM Provider (choose: anthropic, openai, or ollama)
    LLM_PROVIDER: Literal["anthropic", "openai", "ollama"] = "anthropic"
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: str = ""
    
    # OpenAI Configuration
    OPENAI_API_KEY: str = ""
    
    # Ollama Configuration (local, free)
    OLLAMA_URL: str = "http://localhost:11434/api/generate"
    OLLAMA_MODEL: str = "llama3.2"
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Vector Database
    VECTOR_DB_PATH: str = "./data/processed/chromadb"
    COLLECTION_NAME: str = "clinic_faq"
    
    # RAG Configuration
    TOP_K_RESULTS: int = 3
    SIMILARITY_THRESHOLD: float = 0.6
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    
    # Clinic Info
    CLINIC_NAME: str = "HealthCare Plus Clinic"
    CLINIC_PHONE: str = "+1-555-123-4567"
    CLINIC_EMAIL: str = "info@healthcareplus.com"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()