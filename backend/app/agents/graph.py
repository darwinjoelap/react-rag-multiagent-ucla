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
    
    Lógica:
    1. Si hay documentos → answer
    2. Si retry_count >= 2 → answer (forzar aunque no haya docs)
    3. Si no → rewrite
    """
    relevant_docs = state.get("retrieved_documents", [])
    retry_count = state.get("retry_count", 0)
    
    # Si hay documentos, proceder a respuesta
    if len(relevant_docs) > 0:
        logger.info(f"Grader: {len(relevant_docs)} docs → answer")
        return "answer"
    
    # Si alcanzó límite de reintentos, forzar respuesta
    if retry_count >= 2:
        logger.warning(f"Grader: Límite de reintentos alcanzado ({retry_count}) → answer forzado")
        return "answer"
    
    # Intentar reescribir
    logger.info(f"Grader: 0 docs, retry {retry_count}/2 → rewrite")
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


# ==================== STREAMING SUPPORT ====================

import json
from datetime import datetime
from typing import AsyncGenerator


async def stream_graph(user_query: str) -> AsyncGenerator[str, None]:
    """
    Ejecutar el grafo con streaming de eventos
    
    Emite eventos en formato Server-Sent Events (SSE) durante la ejecución:
    - node_start: Cuando inicia un nodo
    - node_end: Cuando termina un nodo
    - thought: Pensamiento del coordinador
    - documents_retrieved: Documentos recuperados
    - grading_result: Resultado del grader
    - rewrite: Query reescrita
    - final_answer: Respuesta final
    - error: Error durante procesamiento
    - done: Procesamiento completado
    
    Args:
        user_query: Pregunta del usuario
        
    Yields:
        Eventos en formato SSE: "data: {json}\n\n"
    """
    from app.agents.state import create_initial_state
    import time
    
    logger.info(f"🌊 Iniciando streaming para query: '{user_query}'")
    start_time = time.time()
    
    try:
        # Crear estado inicial
        initial_state = create_initial_state(user_query)
        
        # Obtener grafo
        graph = get_graph()
        
        # Variable para rastrear nodo actual
        current_node = None
        
        # Streaming del grafo
        async for event in graph.astream(initial_state):
            try:
                # El evento viene como dict con el nombre del nodo como key
                node_name = list(event.keys())[0] if event else None
                node_state = event.get(node_name, {}) if node_name else {}
                
                logger.debug(f"📦 Evento recibido: {node_name}")
                
                # Emitir evento de inicio de nodo (si cambió)
                if node_name and node_name != current_node:
                    current_node = node_name
                    
                    event_data = {
                        "event_type": "node_start",
                        "node_name": node_name,
                        "iteration": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(event_data)}\n\n"
                
                # Emitir eventos específicos según el nodo
                
                # COORDINATOR: Emitir pensamiento y acción
                if node_name == "coordinator" and "thought" in node_state:
                    thought_event = {
                        "event_type": "thought",
                        "thought": node_state.get("thought", ""),
                        "action": node_state.get("action", ""),
                        "iteration": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(thought_event)}\n\n"
                
                # SEARCH: Emitir documentos recuperados
                if node_name == "search" and "retrieved_documents" in node_state:
                    docs = node_state.get("retrieved_documents", [])
                    sources = list(set([
                        doc.get("metadata", {}).get("source", "unknown").split("\\")[-1] 
                        for doc in docs
                    ]))
                    
                    docs_event = {
                        "event_type": "documents_retrieved",
                        "document_count": len(docs),
                        "sources": sources,
                        "iteration": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(docs_event)}\n\n"
                
                # GRADER: Emitir resultado de evaluación
                if node_name == "grader" and "retrieved_documents" in node_state:
                    docs = node_state.get("retrieved_documents", [])
                    total = len(docs)
                    
                    grading_event = {
                        "event_type": "grading_result",
                        "relevant_count": total,  # Todos se consideran relevantes por ahora
                        "total_count": total,
                        "decision": "proceed" if total > 0 else "rewrite",
                        "iteration": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(grading_event)}\n\n"
                
                # REWRITER: Emitir query reescrita
                if node_name == "rewriter" and "rewritten_query" in node_state:
                    rewrite_event = {
                        "event_type": "rewrite",
                        "original_query": user_query,
                        "rewritten_query": node_state.get("rewritten_query", ""),
                        "iteration": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(rewrite_event)}\n\n"
                
                # ANSWER: Emitir respuesta final
                if node_name == "answer" and "final_answer" in node_state:
                    # Formatear fuentes
                    sources = []
                    for doc in node_state.get("retrieved_documents", []):
                        sources.append({
                            "document": doc.get("document", "")[:200] + "...",
                            "source": doc.get("metadata", {}).get("source", "unknown").split("\\")[-1],
                            "similarity": round(doc.get("similarity", 0.0), 4)
                        })
                    
                    answer_event = {
                        "event_type": "final_answer",
                        "answer": node_state.get("final_answer", ""),
                        "sources": sources,
                        "total_iterations": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(answer_event)}\n\n"
                
                # Emitir evento de fin de nodo
                if node_name:
                    end_event = {
                        "event_type": "node_end",
                        "node_name": node_name,
                        "iteration": node_state.get("iteration", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    yield f"data: {json.dumps(end_event)}\n\n"
                    
            except Exception as e:
                logger.error(f"❌ Error procesando evento de nodo: {str(e)}")
                error_event = {
                    "event_type": "error",
                    "error_message": str(e),
                    "node_name": node_name,
                    "timestamp": datetime.now().isoformat()
                }
                yield f"data: {json.dumps(error_event)}\n\n"
        
        # Emitir evento de finalización
        elapsed_time = time.time() - start_time
        done_event = {
            "event_type": "done",
            "success": True,
            "total_time_seconds": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(done_event)}\n\n"
        
        logger.info(f"✅ Streaming completado en {elapsed_time:.2f}s")
        
    except Exception as e:
        logger.error(f"❌ Error fatal en streaming: {str(e)}", exc_info=True)
        
        # Emitir evento de error fatal
        error_event = {
            "event_type": "error",
            "error_message": f"Error fatal: {str(e)}",
            "node_name": None,
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(error_event)}\n\n"
        
        # Emitir done con error
        elapsed_time = time.time() - start_time
        done_event = {
            "event_type": "done",
            "success": False,
            "total_time_seconds": round(elapsed_time, 2),
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(done_event)}\n\n"