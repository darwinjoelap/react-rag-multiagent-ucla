"""
Router para endpoints de documentos
"""
from fastapi import APIRouter, HTTPException, status
import logging
from datetime import datetime

from app.models.schemas import DocumentStats

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/stats", response_model=DocumentStats)
async def get_stats():
    """
    Obtener estadísticas del vector store
    
    Returns:
    - Total de documentos indexados
    - Número de archivos fuente únicos
    - Distribución por tipo de archivo
    - Timestamp de última actualización
    """
    try:
        from app.services.vector_store import VectorStoreService
        
        logger.info("📊 Obteniendo estadísticas del vector store...")
        
        vs = VectorStoreService()
        stats = vs.get_stats()
        
        return DocumentStats(
            total_documents=stats["total_documents"],
            unique_sources=stats["unique_sources"],
            file_types=stats.get("file_types", {}),
            last_updated=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"❌ Error obteniendo estadísticas: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error obteniendo estadísticas: {str(e)}"
        )


@router.get("/sources")
async def list_sources():
    """
    Listar todos los archivos fuente indexados
    
    Returns:
    - Lista de nombres de archivos
    - Número de chunks por archivo
    """
    try:
        from app.services.vector_store import VectorStoreService
        
        vs = VectorStoreService()
        
        # Obtener todos los metadatos
        results = vs.collection.get(
            include=["metadatas"]
        )
        
        # Agrupar por fuente
        sources = {}
        for metadata in results["metadatas"]:
            source = metadata.get("source", "unknown")
            filename = source.split("\\")[-1]  # Solo nombre del archivo
            
            if filename not in sources:
                sources[filename] = {
                    "filename": filename,
                    "full_path": source,
                    "chunk_count": 0
                }
            sources[filename]["chunk_count"] += 1
        
        return {
            "total_sources": len(sources),
            "sources": list(sources.values())
        }
        
    except Exception as e:
        logger.error(f"❌ Error listando fuentes: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listando fuentes: {str(e)}"
        )


@router.post("/reindex")
async def reindex_documents():
    """
    Re-indexar documentos desde data/raw/
    
    ADVERTENCIA: Esto eliminará el índice actual y lo recreará
    """
    try:
        from app.services.document_loader import DocumentLoader
        from app.services.vector_store import VectorStoreService
        
        logger.info("🔄 Iniciando re-indexación de documentos...")
        
        # 1. Eliminar colección actual
        vs = VectorStoreService()
        vs.delete_collection()
        logger.info("✅ Colección anterior eliminada")
        
        # 2. Cargar documentos
        loader = DocumentLoader()
        documents = loader.load_directory("../data/raw")
        logger.info(f"✅ {len(documents)} documentos cargados")
        
        # 3. Re-indexar
        result = vs.add_documents(documents)
        logger.info(f"✅ {result['added']} documentos indexados")
        
        return {
            "message": "Re-indexación completada",
            "documents_indexed": result["added"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error en re-indexación: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error en re-indexación: {str(e)}"
        )