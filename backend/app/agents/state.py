"""
Estado del Grafo con Soporte Multi-Turno
"""

from typing import TypedDict, List, Dict, Any, Optional
from datetime import datetime


class Message(TypedDict):
    """Mensaje individual en la conversación"""
    role: str  # "user" o "assistant"
    content: str
    timestamp: str


class GraphState(TypedDict):
    """
    Estado del grafo RAG con historial de conversación
    """
    # Conversación
    messages: List[Message]  # ← NUEVO: Historial completo
    current_query: str  # Consulta actual del usuario
    
    # Documentos
    retrieved_documents: List[Dict[str, Any]]
    
    # Control de flujo
    iteration: int
    retry_count: int  # Contador de rewrites
    next_step: str
    should_continue: bool
    
    # ReAct
    thought: str
    action: str
    action_input: str
    
    # Resultados
    final_answer: str
    rewritten_query: Optional[str]
    
    # Trace/Debug
    trace: List[Dict[str, Any]]
    error: Optional[str]


def create_initial_state(user_query: str, conversation_history: List[Message] = None) -> GraphState:
    """
    Crear estado inicial con historial opcional
    
    Args:
        user_query: Consulta actual del usuario
        conversation_history: Historial previo de la conversación (opcional)
    """
    # Si no hay historial, crear lista vacía
    if conversation_history is None:
        conversation_history = []
    
    # Agregar mensaje actual del usuario
    messages = conversation_history + [{
        "role": "user",
        "content": user_query,
        "timestamp": datetime.now().isoformat()
    }]
    
    return {
        "messages": messages,
        "current_query": user_query,
        "retrieved_documents": [],
        "iteration": 1,
        "retry_count": 0,
        "next_step": "coordinator",
        "should_continue": True,
        "thought": "",
        "action": "",
        "action_input": "",
        "final_answer": "",
        "rewritten_query": None,
        "trace": [],
        "error": None
    }


def add_assistant_message(state: GraphState, content: str) -> Dict:
    """
    Agregar mensaje del asistente al historial
    """
    new_message = {
        "role": "assistant",
        "content": content,
        "timestamp": datetime.now().isoformat()
    }
    
    return {
        "messages": state["messages"] + [new_message]
    }


def get_conversation_context(state: GraphState, last_n: int = 5) -> str:
    """
    Obtener contexto de los últimos N mensajes
    
    Args:
        state: Estado actual
        last_n: Número de mensajes a incluir
        
    Returns:
        String con el historial formateado
    """
    messages = state["messages"][-last_n:]
    
    context = "Historial de la conversación:\n\n"
    
    for msg in messages:
        role = "Usuario" if msg["role"] == "user" else "Asistente"
        context += f"{role}: {msg['content']}\n\n"
    
    return context


# Mantener funciones originales para compatibilidad
def increment_iteration(state: GraphState) -> Dict:
    """Incrementar contador de iteración"""
    return {"iteration": state["iteration"] + 1}


def increment_retry(state: GraphState) -> Dict:
    """Incrementar contador de reintentos"""
    return {"retry_count": state["retry_count"] + 1}


def add_trace_step(state: GraphState, agent: str, thought: str, action: str, observation: str) -> Dict:
    """Agregar paso al trace"""
    trace_step = {
        "agent": agent,
        "thought": thought,
        "action": action,
        "observation": observation,
        "timestamp": datetime.now().isoformat()
    }
    
    return {"trace": state["trace"] + [trace_step]}


def should_continue_graph(state: GraphState) -> bool:
    """Verificar si debe continuar el grafo"""
    return state.get("should_continue", False) and state.get("iteration", 0) < 5