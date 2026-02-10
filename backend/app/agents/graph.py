"""
LangGraph Workflow - Grafo completo del sistema RAG Multiagente

Conecta todos los nodos (coordinator, search, grader, rewriter, answer)
en un flujo condicional que implementa el patrón ReAct.
"""

import logging
from typing import Literal
from langgraph.graph import StateGraph, END
from app.agents.state import GraphState, should_continue_graph
from app.agents.coordinator import coordinator_node
from app.agents.search_node import search_node
from app.agents.grader import grader_node
from app.agents.rewriter import rewriter_node
from app.agents.answer_node import answer_node

logger = logging.getLogger(__name__)


def route_after_coordinator(state: GraphState) -> Literal["search", "answer", "end"]:
    """
    Decidir siguiente nodo después del coordinador
    
    Basado en la acción decidida por el coordinador:
    - search → ir a búsqueda
    - answer/clarify → ir a respuesta final
    """
    action = state.get("action", "answer")
    
    if action == "search":
        return "search"
    else:  # answer o clarify
        return "answer"


def route_after_grader(state: GraphState) -> Literal["rewrite", "answer", "end"]:
    """
    Decidir siguiente nodo después del grader
    
    TEMPORAL: Siempre ir a answer si hay documentos
    """
    relevant_docs = state.get("retrieved_documents", [])
    
    # Si hay CUALQUIER documento, ir a answer
    if len(relevant_docs) > 0:
        logger.info(f"Grader: {len(relevant_docs)} docs → answer")
        return "answer"
    else:
        logger.info("Grader: 0 docs → rewrite")
        return "rewrite"


def should_continue_loop(state: GraphState) -> Literal["coordinator", "end"]:
    """
    Decidir si continuar el loop o terminar
    
    Continuar → volver al coordinador
    Terminar → ir a END
    """
    if should_continue_graph(state):
        return "coordinator"
    else:
        return "end"


def build_graph() -> StateGraph:
    """
    Construir el grafo completo del sistema
    
    Flujo:
    1. START → coordinator
    2. coordinator → search | answer
    3. search → grader
    4. grader → answer | rewrite
    5. rewrite → search (volver a buscar)
    6. answer → END
    
    Returns:
        Grafo compilado listo para ejecutar
    """
    logger.info("Construyendo grafo LangGraph")
    
    # Crear grafo
    workflow = StateGraph(GraphState)
    
    # Agregar nodos
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("search", search_node)
    workflow.add_node("grader", grader_node)
    workflow.add_node("rewriter", rewriter_node)
    workflow.add_node("answer", answer_node)
    
    # Definir punto de entrada
    workflow.set_entry_point("coordinator")
    
    # Edges condicionales
    
    # Después del coordinador
    workflow.add_conditional_edges(
        "coordinator",
        route_after_coordinator,
        {
            "search": "search",
            "answer": "answer",
            "end": END
        }
    )
    
    # Después de search, siempre va a grader
    workflow.add_edge("search", "grader")
    
    # Después del grader
    workflow.add_conditional_edges(
        "grader",
        route_after_grader,
        {
            "answer": "answer",
            "rewrite": "rewriter",
            "end": END
        }
    )
    
    # Después de rewriter, volver a search
    workflow.add_edge("rewriter", "search")
    
    # Después de answer, terminar
    workflow.add_edge("answer", END)
    
    # Compilar grafo
    graph = workflow.compile()
    
    logger.info("Grafo compilado exitosamente")
    return graph


# Instancia global del grafo
_graph_instance = None


def get_graph() -> StateGraph:
    """Obtener instancia singleton del grafo"""
    global _graph_instance
    
    if _graph_instance is None:
        _graph_instance = build_graph()
    
    return _graph_instance


def run_graph(user_query: str) -> GraphState:
    """
    Ejecutar el grafo completo con una consulta del usuario
    
    Args:
        user_query: Pregunta del usuario
        
    Returns:
        Estado final después de ejecutar el grafo
    """
    from app.agents.state import create_initial_state
    
    logger.info(f"Ejecutando grafo para query: '{user_query}'")
    
    # Crear estado inicial
    initial_state = create_initial_state(user_query)
    
    # Obtener grafo
    graph = get_graph()
    
    # Ejecutar
    final_state = graph.invoke(initial_state)
    
    logger.info("Grafo ejecutado completamente")
    
    return final_state