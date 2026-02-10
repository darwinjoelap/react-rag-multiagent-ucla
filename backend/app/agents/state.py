from typing import TypedDict, List, Dict, Optional, Literal, Annotated
from datetime import datetime
import operator

class GraphState(TypedDict):
    """
    Estado compartido del grafo de agentes LangGraph
    
    Este estado se pasa entre todos los nodos del grafo y contiene
    toda la información necesaria para el flujo ReAct.
    
    El patrón ReAct sigue: Thought → Action → Observation
    """
    
    # ============= ENTRADA DEL USUARIO =============
    user_query: str
    """Consulta original del usuario"""
    
    # ============= HISTORIAL =============
    conversation_history: Annotated[List[Dict[str, str]], operator.add]
    """
    Historial de mensajes acumulativo
    Formato: [{"role": "user/assistant", "content": "..."}]
    """
    
    # ============= ESTADO REACT =============
    iteration: int
    """Número de iteración actual (máximo 5 para evitar loops)"""
    
    thought: Optional[str]
    """Pensamiento actual del agente (paso 1 de ReAct)"""
    
    action: Optional[str]
    """Acción a ejecutar: 'search', 'answer', 'clarify'"""
    
    action_input: Optional[str]
    """Input específico para la acción"""
    
    observation: Optional[str]
    """Resultado/observación de la última acción ejecutada"""
    
    # ============= CONTEXTO Y DOCUMENTOS =============
    retrieved_documents: Annotated[List[Dict], operator.add]
    """Lista acumulativa de documentos recuperados"""
    
    relevant_context: Optional[str]
    """Contexto concatenado y formateado para el LLM"""
    
    # ============= RESPUESTA FINAL =============
    final_answer: Optional[str]
    """Respuesta final generada para el usuario"""
    
    # ============= CONTROL DE FLUJO =============
    next_step: Literal["coordinator", "search", "answer", "clarify", "end"]
    """Siguiente nodo a ejecutar en el grafo"""
    
    should_continue: bool
    """Flag para continuar o terminar el loop"""
    
    # ============= METADATA Y DEBUGGING =============
    trace: Annotated[List[Dict], operator.add]
    """
    Traza completa ReAct para debugging y visualización
    Cada elemento: {step, agent, thought, action, observation, timestamp}
    """
    
    timestamp: str
    """Timestamp de inicio del proceso"""
    
    error: Optional[str]
    """Mensaje de error si algo falla"""


def create_initial_state(user_query: str) -> GraphState:
    """
    Crear estado inicial del grafo
    
    Args:
        user_query: Consulta del usuario
        
    Returns:
        GraphState inicializado con valores por defecto
    """
    return GraphState(
        # Input
        user_query=user_query,
        
        # Historial
        conversation_history=[
            {"role": "user", "content": user_query}
        ],
        
        # ReAct
        iteration=0,
        thought=None,
        action=None,
        action_input=None,
        observation=None,
        
        # Contexto
        retrieved_documents=[],
        relevant_context=None,
        
        # Respuesta
        final_answer=None,
        
        # Control
        next_step="coordinator",
        should_continue=True,
        
        # Metadata
        trace=[],
        timestamp=datetime.now().isoformat(),
        error=None
    )


def add_trace_step(
    state: GraphState,
    agent: str,
    thought: str,
    action: str,
    observation: str
) -> Dict:
    """
    Agregar un paso a la traza ReAct
    
    Args:
        state: Estado actual del grafo
        agent: Nombre del agente que ejecuta el paso
        thought: Pensamiento/razonamiento del agente
        action: Acción ejecutada
        observation: Resultado observado
        
    Returns:
        Diccionario con el paso de la traza para agregar al estado
    """
    trace_step = {
        "step": state["iteration"],
        "agent": agent,
        "thought": thought,
        "action": action,
        "observation": observation,
        "timestamp": datetime.now().isoformat()
    }
    
    return {"trace": [trace_step]}


def increment_iteration(state: GraphState) -> Dict:
    """
    Incrementar contador de iteraciones
    
    Args:
        state: Estado actual
        
    Returns:
        Diccionario con iteración incrementada
    """
    return {"iteration": state["iteration"] + 1}


def should_continue_graph(state: GraphState) -> bool:
    """
    Determinar si el grafo debe continuar ejecutándose
    
    Condiciones de parada:
    - Se alcanzó el máximo de iteraciones (5)
    - Se generó una respuesta final
    - Hay un error
    
    Args:
        state: Estado actual
        
    Returns:
        True si debe continuar, False si debe terminar
    """
    # Parar si hay error
    if state.get("error"):
        return False
    
    # Parar si hay respuesta final
    if state.get("final_answer"):
        return False
    
    # Parar si se alcanzó máximo de iteraciones
    if state.get("iteration", 0) >= 5:
        return False
    
    return state.get("should_continue", True)