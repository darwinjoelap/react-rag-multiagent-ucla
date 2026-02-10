from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from app.services.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

class VectorStoreService:
    """Servicio para manejar el almacenamiento vectorial con ChromaDB"""
    
    def __init__(
        self,
        persist_directory: str = "../data/vectorstore",
        collection_name: str = "ucla_documents",
        embedding_service: Optional[EmbeddingService] = None
    ):
        """
        Inicializar servicio de vector store
        
        Args:
            persist_directory: Directorio donde se persiste ChromaDB
            collection_name: Nombre de la colección
            embedding_service: Servicio de embeddings (si no se provee, crea uno nuevo)
        """
        self.persist_directory = Path(persist_directory).resolve()
        self.collection_name = collection_name
        
        # Crear directorio si no existe
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Inicializar ChromaDB client
        logger.info(f"Inicializando ChromaDB en: {self.persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Servicio de embeddings
        self.embedding_service = embedding_service or EmbeddingService()
        
        # Obtener o crear colección
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Vector store inicializado. Documentos en colección: {self.collection.count()}")
    
    def _get_or_create_collection(self):
        """Obtener colección existente o crear una nueva"""
        try:
            # PRIMERO: Intentar obtener por NOMBRE (no por ID)
            collection = self.client.get_collection(
                name=self.collection_name
            )
            logger.info(f"Colección '{self.collection_name}' cargada con {collection.count()} docs")
            return collection
        except Exception as e:
            # Si no existe, crear nueva
            logger.info(f"Creando nueva colección '{self.collection_name}'")
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "UCLA RAG Multiagent Knowledge Base"}
            )
            return collection
    
    def add_documents(
        self,
        documents: List[Document],
        batch_size: int = 100
    ) -> Dict[str, Any]:
        """
        Agregar documentos a la colección
        
        Args:
            documents: Lista de documentos LangChain
            batch_size: Tamaño de lote para procesamiento
            
        Returns:
            Diccionario con estadísticas de la operación
        """
        if not documents:
            logger.warning("No hay documentos para agregar")
            return {"added": 0, "total": 0}
        
        logger.info(f"Agregando {len(documents)} documentos a la colección")
        
        # Preparar datos
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        # Generar IDs únicos
        existing_count = self.collection.count()
        ids = [f"doc_{existing_count + i}" for i in range(len(documents))]
        
        # Generar embeddings
        logger.info("Generando embeddings...")
        embeddings = self.embedding_service.embed_texts(texts)
        
        # Agregar en lotes
        total_added = 0
        for i in range(0, len(documents), batch_size):
            end_idx = min(i + batch_size, len(documents))
            
            batch_ids = ids[i:end_idx]
            batch_embeddings = embeddings[i:end_idx]
            batch_texts = texts[i:end_idx]
            batch_metadatas = metadatas[i:end_idx]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_texts,
                metadatas=batch_metadatas
            )
            
            total_added += len(batch_ids)
            logger.info(f"Agregados {total_added}/{len(documents)} documentos")
        
        result = {
            "added": total_added,
            "total": self.collection.count(),
            "collection": self.collection_name
        }
        
        logger.info(f"✅ Documentos agregados exitosamente: {result}")
        return result
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos similares a la consulta
        
        Args:
            query: Texto de búsqueda
            n_results: Número de resultados a devolver
            filter_metadata: Filtros de metadata (opcional)
            
        Returns:
            Lista de documentos con scores de similitud
        """
        logger.info(f"Buscando: '{query}' (top {n_results})")
        
        # Generar embedding de la consulta
        query_embedding = self.embedding_service.embed_query(query)
        
        # Buscar en ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Formatear resultados
        formatted_results = []
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                "id": results['ids'][0][i],
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i],
                "similarity": 1 - results['distances'][0][i]  # Convertir distancia a similitud
            })
        
        logger.info(f"Encontrados {len(formatted_results)} resultados")
        return formatted_results
    
    def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Obtener documento por ID"""
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas", "embeddings"]
            )
            
            if result['ids']:
                return {
                    "id": result['ids'][0],
                    "document": result['documents'][0],
                    "metadata": result['metadatas'][0],
                    "embedding": result['embeddings'][0] if result['embeddings'] else None
                }
            return None
        except Exception as e:
            logger.error(f"Error obteniendo documento {doc_id}: {str(e)}")
            return None
    
    def delete_by_id(self, doc_id: str) -> bool:
        """Eliminar documento por ID"""
        try:
            self.collection.delete(ids=[doc_id])
            logger.info(f"Documento {doc_id} eliminado")
            return True
        except Exception as e:
            logger.error(f"Error eliminando documento {doc_id}: {str(e)}")
            return False
    
    def delete_collection(self):
        """Eliminar toda la colección (usar con precaución)"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.warning(f"Colección '{self.collection_name}' eliminada")
            # Recrear colección vacía
            self.collection = self._get_or_create_collection()
        except Exception as e:
            logger.error(f"Error eliminando colección: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas de la colección"""
        count = self.collection.count()
        
        stats = {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": str(self.persist_directory),
            "embedding_dimension": self.embedding_service.dimension
        }
        
        # Obtener tipos de archivos si hay documentos
        if count > 0:
            sample = self.collection.get(
                limit=min(100, count),
                include=["metadatas"]
            )
            
            file_types = {}
            sources = set()
            
            for metadata in sample['metadatas']:
                ftype = metadata.get('file_type', 'unknown')
                file_types[ftype] = file_types.get(ftype, 0) + 1
                
                source = metadata.get('source', '')
                if source:
                    sources.add(source)
            
            stats['file_types'] = file_types
            stats['unique_sources'] = len(sources)
        
        return stats
    
    def search_by_metadata(
        self,
        metadata_filter: Dict,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Buscar documentos por metadata
        
        Args:
            metadata_filter: Filtro de metadata (ej: {"file_type": "pdf"})
            limit: Número máximo de resultados
            
        Returns:
            Lista de documentos que coinciden
        """
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=limit,
                include=["documents", "metadatas"]
            )
            
            formatted_results = []
            for i in range(len(results['ids'])):
                formatted_results.append({
                    "id": results['ids'][i],
                    "document": results['documents'][i],
                    "metadata": results['metadatas'][i]
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error buscando por metadata: {str(e)}")
            return []

# Instancia global (singleton)
_vector_store_instance = None

def get_vector_store(
    persist_directory: str = "../data/vectorstore",
    collection_name: str = "ucla_documents"
) -> VectorStoreService:
    """Obtener instancia singleton del vector store"""
    global _vector_store_instance
    
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreService(
            persist_directory=persist_directory,
            collection_name=collection_name
        )
    
    return _vector_store_instance