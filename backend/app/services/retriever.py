from typing import List, Dict, Optional, Any
import logging
from app.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

class RetrieverService:
    """Servicio de recuperación semántica de documentos"""
    
    def __init__(
        self,
        vector_store: Optional[VectorStoreService] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Inicializar servicio de recuperación
        
        Args:
            vector_store: Instancia del vector store (si no se provee, crea una nueva)
            top_k: Número de documentos a recuperar por defecto
            similarity_threshold: Umbral mínimo de similitud (0-1)
        """
        self.vector_store = vector_store or VectorStoreService()
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        
        logger.info(f"Retriever inicializado (top_k={top_k}, threshold={similarity_threshold})")
    
    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict] = None,
        min_similarity: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Recuperar documentos relevantes para una consulta
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados (usa default si no se especifica)
            filter_metadata: Filtros de metadata (ej: {"file_type": "pdf"})
            min_similarity: Similitud mínima (usa default si no se especifica)
            
        Returns:
            Lista de documentos relevantes con scores
        """
        k = top_k or self.top_k
        threshold = min_similarity or self.similarity_threshold
        
        logger.info(f"Recuperando documentos para: '{query}' (top_k={k}, threshold={threshold})")
        
        # Buscar en vector store
        results = self.vector_store.search(
            query=query,
            n_results=k * 2,  # Buscar el doble para filtrar después
            filter_metadata=filter_metadata
        )
        
        # Filtrar por umbral de similitud
        filtered_results = [
            result for result in results
            if result['similarity'] >= threshold
        ]
        
        # Limitar a top_k
        final_results = filtered_results[:k]
        
        logger.info(f"Recuperados {len(final_results)} documentos (filtrados de {len(results)})")
        
        return final_results
    
    def retrieve_with_context(
        self,
        query: str,
        top_k: Optional[int] = None,
        context_window: int = 1
    ) -> Dict[str, Any]:
        """
        Recuperar documentos con contexto adicional
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados
            context_window: Número de chunks vecinos a incluir (antes/después)
            
        Returns:
            Diccionario con resultados y contexto
        """
        results = self.retrieve(query, top_k)
        
        # TODO: Implementar recuperación de chunks vecinos
        # Por ahora retorna los resultados básicos
        
        return {
            "query": query,
            "results": results,
            "total_found": len(results),
            "context_window": context_window
        }
    
    def retrieve_by_source(
        self,
        query: str,
        source_file: str,
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Recuperar documentos de un archivo específico
        
        Args:
            query: Consulta del usuario
            source_file: Nombre del archivo fuente
            top_k: Número de resultados
            
        Returns:
            Lista de documentos del archivo especificado
        """
        # Buscar con filtro de source
        results = self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata={"source": {"$contains": source_file}}
        )
        
        logger.info(f"Recuperados {len(results)} documentos de '{source_file}'")
        return results
    
    def retrieve_diverse(
        self,
        query: str,
        top_k: Optional[int] = None,
        diversity_factor: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Recuperar documentos diversos (evitar resultados muy similares entre sí)
        
        Args:
            query: Consulta del usuario
            top_k: Número de resultados
            diversity_factor: Factor de diversidad (0=solo similitud, 1=máxima diversidad)
            
        Returns:
            Lista de documentos diversos
        """
        k = top_k or self.top_k
        
        # Recuperar más documentos de lo necesario
        candidates = self.retrieve(query, top_k=k * 3)
        
        if not candidates:
            return []
        
        # Algoritmo simple de diversificación
        # Seleccionar el primero (más relevante)
        selected = [candidates[0]]
        candidates_remaining = candidates[1:]
        
        while len(selected) < k and candidates_remaining:
            # Para cada candidato restante, calcular similitud mínima con seleccionados
            best_candidate = None
            best_score = -1
            
            for candidate in candidates_remaining:
                # Score combinado: relevancia + diversidad
                relevance_score = candidate['similarity']
                
                # Calcular diversidad (simplificado: diferencia de contenido)
                diversity_score = 1.0  # Placeholder
                
                combined_score = (
                    (1 - diversity_factor) * relevance_score +
                    diversity_factor * diversity_score
                )
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate
            
            if best_candidate:
                selected.append(best_candidate)
                candidates_remaining.remove(best_candidate)
        
        logger.info(f"Recuperados {len(selected)} documentos diversos")
        return selected
    
    def get_relevant_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Obtener contexto relevante como texto concatenado
        Útil para prompts de LLM
        
        Args:
            query: Consulta del usuario
            max_tokens: Máximo de tokens aproximados (caracteres / 4)
            
        Returns:
            Contexto concatenado
        """
        # Estimar caracteres basado en tokens
        max_chars = max_tokens * 4
        
        results = self.retrieve(query, top_k=10)
        
        context_parts = []
        total_chars = 0
        
        for i, result in enumerate(results, 1):
            doc_text = result['document']
            source = result['metadata'].get('source', 'Unknown')
            
            # Formatear chunk con metadata
            chunk_text = f"[Documento {i} - {source}]\n{doc_text}\n"
            
            # Verificar si cabe
            if total_chars + len(chunk_text) > max_chars:
                break
            
            context_parts.append(chunk_text)
            total_chars += len(chunk_text)
        
        context = "\n---\n".join(context_parts)
        
        logger.info(f"Contexto generado: {len(context_parts)} chunks, {total_chars} caracteres")
        return context
    
    def analyze_query_coverage(
        self,
        query: str,
        top_k: int = 10
    ) -> Dict[str, Any]:
        """
        Analizar qué tan bien la base de conocimiento cubre la consulta
        
        Args:
            query: Consulta a analizar
            top_k: Número de documentos a considerar
            
        Returns:
            Análisis de cobertura
        """
        results = self.retrieve(query, top_k=top_k)
        
        if not results:
            return {
                "query": query,
                "coverage": "none",
                "avg_similarity": 0.0,
                "max_similarity": 0.0,
                "sources_found": 0
            }
        
        similarities = [r['similarity'] for r in results]
        sources = set(r['metadata'].get('source', 'Unknown') for r in results)
        
        avg_sim = sum(similarities) / len(similarities)
        max_sim = max(similarities)
        
        # Determinar nivel de cobertura
        if max_sim >= 0.9:
            coverage = "excellent"
        elif max_sim >= 0.8:
            coverage = "good"
        elif max_sim >= 0.7:
            coverage = "fair"
        else:
            coverage = "poor"
        
        return {
            "query": query,
            "coverage": coverage,
            "avg_similarity": round(avg_sim, 4),
            "max_similarity": round(max_sim, 4),
            "min_similarity": round(min(similarities), 4),
            "sources_found": len(sources),
            "total_chunks": len(results)
        }
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: Optional[int] = None
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Recuperar documentos para múltiples consultas
        
        Args:
            queries: Lista de consultas
            top_k: Número de resultados por consulta
            
        Returns:
            Diccionario {query: resultados}
        """
        logger.info(f"Procesamiento batch de {len(queries)} consultas")
        
        results = {}
        for query in queries:
            results[query] = self.retrieve(query, top_k=top_k)
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtener estadísticas del retriever y vector store"""
        vs_stats = self.vector_store.get_stats()
        
        return {
            **vs_stats,
            "default_top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }

# Instancia global
_retriever_instance = None

def get_retriever(
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> RetrieverService:
    """Obtener instancia singleton del retriever"""
    global _retriever_instance
    
    if _retriever_instance is None:
        _retriever_instance = RetrieverService(
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
    
    return _retriever_instance