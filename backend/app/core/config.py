from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # Application
    APP_NAME: str = "RAG Multiagent System"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"
    
    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_PREFIX: str = "/api/v1"
    
    # CORS
    ALLOWED_ORIGINS: str = "http://localhost:5173,http://localhost:3000"
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2:latest"
    
    # ChromaDB
    CHROMA_PERSIST_DIRECTORY: str = "../data/vectorstore"
    CHROMA_COLLECTION_NAME: str = "ucla_documents"
    
    # Document Processing
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_DOCUMENTS: int = 1000
    
    # Embeddings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    
    # Agent Configuration
    MAX_ITERATIONS: int = 5
    AGENT_TIMEOUT: int = 300
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"  # ← AGREGAR ESTA LÍNEA
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # ← AGREGAR ESTA LÍNEA (ignora campos extra del .env)

@lru_cache()
def get_settings() -> Settings:
    return Settings()

settings = get_settings()