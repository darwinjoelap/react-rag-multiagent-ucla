"""
Search Node - Busca documentos en la base de conocimiento

Utiliza el retriever service para encontrar documentos relevantes
basados en la query del coordinador o rewriter.
"""

import logging
from typing import Dict
from app.agents.state import GraphState, add_trace_step
from app.services.retriever import RetrieverService

logger = logging.getLogger(__name__)


class SearchNode:
    """
    Nodo que ejecuta búsqueda semántica en el vector store
    """
    
    def __init__(self):
        """Inicializar con retriever service"""
        self.retriever = RetrieverService(
            top_k=5,
            similarity_threshold=0.2
        )
        logger.info("SearchNode inicializado")
    
    def __call__(self, state: GraphState) -> Dict:
        """
        Ejecutar búsqueda de documentos
        """
        logger.info("Search ejecutando - Buscando documentos")
    
        try:
            # Obtener query de búsqueda
            search_query = state.get("action_input", state.get("user_query"))
        
            if not search_query:
                logger.warning("No hay query de búsqueda")
                return {
                    "observation": "No search query provided",
                    "next_step": "answer",
                    **add_trace_step(
                        state,
                        agent="search",
                        thought="Sin query de búsqueda",
                        action="skip",
                        observation="No query to search"
                    )
                }
        
            # DEBUGGING
            logger.info(f"Buscando: '{search_query}'")
            print(f"🔍 Search query: '{search_query}'")
        
            # Ejecutar búsqueda
            results = self.retriever.retrieve(
                query=search_query,
                top_k=5
            )
        
            # DEBUGGING
            print(f"📊 Resultados encontrados: {len(results)}")
            logger.info(f"Encontrados {len(results)} documentos")
        
            # Formatear resultados
            documents = []
            for result in results:
                documents.append({
                    "document": result["document"],
                    "metadata": result["metadata"],
                    "similarity": result["similarity"]
                })
                print(f"  ✅ Doc: sim={result['similarity']:.4f}")
        
            # Construir contexto
            context_parts = []
            for i, doc in enumerate(documents, 1):
                source = doc["metadata"].get("source", "Unknown")
                context_parts.append(f"[Documento {i} - {source}]\n{doc['document']}\n")
        
            relevant_context = "\n---\n".join(context_parts) if context_parts else ""
        
            observation = f"Búsqueda completada: {len(documents)} documentos encontrados"
        
            return {
                "retrieved_documents": documents,
                "relevant_context": relevant_context,
                "observation": observation,
                "next_step": "grader",
                **add_trace_step(
                    state,
                    agent="search",
                    thought=f"Ejecutando búsqueda para: {search_query}",
                    action="search",
                    observation=observation
                )
            }
        
        except Exception as e:
            logger.error(f"Error en search: {str(e)}")
            print(f"❌ Error en search: {str(e)}")
            return {
                "error": f"Error en search: {str(e)}",
                "next_step": "answer",
                **add_trace_step(
                    state,
                    agent="search",
                    thought="Error al buscar documentos",
                    action="error",
                    observation=str(e)
                )
            }

# Instancia global
search_node_instance = SearchNode()


def search_node(state: GraphState) -> Dict:
    """
    Función wrapper para usar en LangGraph
    
    Args:
        state: Estado del grafo
        
    Returns:
        Updates del estado
    """
    return search_node_instance(state)