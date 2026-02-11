"""
FastAPI application para sistema RAG Multiagente UCLA
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from datetime import datetime
import logging
import os

from app.routers import chat, documents

# Cargar variables de entorno
load_dotenv()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Crear aplicación
app = FastAPI(
    title="RAG Multiagent System - UCLA",
    version="1.0.0",
    description="Sistema RAG Multiagente con LangGraph para análisis académico",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])


@app.get("/", tags=["Root"])
async def root():
    """Endpoint raíz - Mensaje de bienvenida"""
    return {
        "message": "RAG Multiagent System API",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint
    
    Verifica el estado de:
    - API FastAPI
    - Ollama (LLM)
    - ChromaDB (Vector Store)
    """
    try:
        # Verificar ChromaDB
        from app.services.vector_store import VectorStoreService
        vs = VectorStoreService()
        doc_count = vs.collection.count()
        chromadb_status = "connected"
    except Exception as e:
        logger.error(f"ChromaDB error: {e}")
        chromadb_status = "disconnected"
        doc_count = 0
    
    try:
        # Verificar Ollama
        from app.services.llm import get_llm
        llm = get_llm()
        # Test simple
        llm.invoke("test")
        ollama_status = "connected"
    except Exception as e:
        logger.error(f"Ollama error: {e}")
        ollama_status = "disconnected"
    
    status = "healthy" if chromadb_status == "connected" and ollama_status == "connected" else "degraded"
    
    return {
        "status": status,
        "ollama_status": ollama_status,
        "chromadb_status": chromadb_status,
        "documents_indexed": doc_count,
        "timestamp": datetime.now().isoformat(),
        "environment": os.getenv("ENVIRONMENT", "development")
    }


@app.get("/api/test", tags=["Test"])
async def test_endpoint():
    """Endpoint de prueba del sistema multiagente"""
    try:
        from app.agents.graph import run_graph
        
        # Test simple
        logger.info("🧪 Ejecutando test del grafo...")
        final_state = run_graph("Hola")
        
        return {
            "message": "Sistema multiagente funcionando correctamente",
            "test_query": "Hola",
            "test_response": final_state.get("final_answer", "")[:100] + "...",
            "iterations": final_state.get("iteration", 0),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error en test: {e}")
        return {
            "message": "Error en test del sistema",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@app.on_event("startup")
async def startup_event():
    """Eventos al iniciar la aplicación"""
    logger.info("=" * 70)
    logger.info("🚀 Iniciando RAG Multiagent API - UCLA")
    logger.info("=" * 70)
    logger.info("📚 Verificando Vector Store...")
    
    try:
        from app.services.vector_store import VectorStoreService
        vs = VectorStoreService()
        stats = vs.get_stats()
        logger.info(f"✅ ChromaDB conectado: {stats['total_documents']} documentos indexados")
    except Exception as e:
        logger.error(f"❌ Error conectando ChromaDB: {e}")
    
    logger.info("🤖 Verificando Ollama...")
    try:
        from app.services.llm import get_llm
        llm = get_llm()
        logger.info("✅ Ollama conectado")
    except Exception as e:
        logger.error(f"❌ Error conectando Ollama: {e}")
    
    logger.info("=" * 70)
    logger.info("✅ API iniciada correctamente")
    logger.info("📖 Documentación: http://localhost:8000/docs")
    logger.info("=" * 70)


@app.on_event("shutdown")
async def shutdown_event():
    """Eventos al cerrar la aplicación"""
    logger.info("=" * 70)
    logger.info("👋 Cerrando RAG Multiagent API")
    logger.info("=" * 70)