"""
Coordinador (Agent Router) - Nodo principal del grafo LangGraph

El coordinador analiza cada consulta y decide:
1. search - Buscar en base de conocimiento
2. answer - Responder directamente
"""

import logging
from typing import Dict
from app.agents.state import GraphState, add_trace_step, increment_iteration
from app.agents.prompts import format_coordinator_prompt, parse_react_response
from app.core.llm_config import get_coordinator_llm  # ← NUEVO: Importar LLM optimizado

logger = logging.getLogger(__name__)


class CoordinatorAgent:
    """
    Agente coordinador que decide la estrategia para cada consulta
    
    Implementa el patrón ReAct:
    - Thought: Analiza la situación
    - Action: Decide qué hacer (search/answer)
    - Observation: Procesará el resultado en el siguiente nodo
    """
    
    def __init__(self):
        """Inicializar coordinador con LLM optimizado"""
        self.llm = get_coordinator_llm()  # ← OPTIMIZADO: llama3.2:1b con temp=0.0, tokens=256
        logger.info("CoordinatorAgent inicializado con llama3.2:1b optimizado")
    
    def __call__(self, state: GraphState) -> Dict:
        """
        Ejecutar nodo coordinador
        
        Args:
            state: Estado actual del grafo
            
        Returns:
            Diccionario con updates al estado
        """
        logger.info(f"Coordinador ejecutando - Iteración {state['iteration']}")
        
        try:
            # 1. Generar prompt con contexto
            prompt = format_coordinator_prompt(state)
            
            # 2. Obtener decisión del LLM
            logger.debug("Enviando prompt al LLM")
            response = self.llm.invoke(prompt)
            logger.debug(f"Respuesta del LLM: {response[:200]}...")
            
            # 3. Parsear respuesta ReAct
            parsed = parse_react_response(response)
            thought = parsed["thought"]
            action = parsed["action"]
            action_input = parsed["action_input"]
            
            # ========== MODIFICADO: NO FORZAR BÚSQUEDA ==========
            # El LLM decide si buscar o responder directamente
            # Solo forzamos si estamos en iteración 2+ y no hay documentos
            
            iteration = state.get("iteration", 1)
            has_docs = len(state.get("retrieved_documents", [])) > 0
            retry_count = state.get("retry_count", 0)
            
            # Si ya buscamos 2 veces y no hay documentos, forzar respuesta
            if retry_count >= 2 and not has_docs and action == "search":
                logger.warning("⚠️ Límite de búsquedas alcanzado, forzando respuesta")
                action = "answer"
                thought = f"{thought} [Nota: Respondiendo con conocimiento general después de {retry_count} búsquedas sin resultados]"
            # ====================================================
            
            logger.info(f"Decisión: {action} | Thought: {thought[:50]}...")
            
            # 4. Validar iteraciones máximas
            if iteration >= 5:
                logger.warning("Máximo de iteraciones alcanzado")
                return {
                    "thought": "Máximo de iteraciones alcanzado",
                    "action": "answer",
                    "action_input": "He alcanzado el límite de iteraciones. Con la información disponible, puedo decir que: " + action_input,
                    "next_step": "answer",
                    "should_continue": False,
                    **increment_iteration(state),
                    **add_trace_step(
                        state,
                        agent="coordinator",
                        thought="Límite de iteraciones",
                        action="answer",
                        observation="Forzando respuesta final"
                    )
                }
            
            # 5. Determinar siguiente nodo según la acción
            next_step_map = {
                "search": "search",
                "answer": "answer"
            }
            next_step = next_step_map.get(action, "answer")
            
            # 6. Construir updates del estado
            updates = {
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "next_step": next_step,
                "should_continue": action != "answer",  # Continuar si no es respuesta final
                **increment_iteration(state),
                **add_trace_step(
                    state,
                    agent="coordinator",
                    thought=thought,
                    action=action,
                    observation=f"Decidido: {action}"
                )
            }
            
            logger.info(f"Coordinador completado - Siguiente: {next_step}")
            return updates
            
        except Exception as e:
            logger.error(f"Error en coordinador: {str(e)}")
            return {
                "error": f"Error en coordinador: {str(e)}",
                "next_step": "end",
                "should_continue": False,
                **add_trace_step(
                    state,
                    agent="coordinator",
                    thought="Error fatal",
                    action="error",
                    observation=str(e)
                )
            }


# Instancia global
coordinator_agent = CoordinatorAgent()


def coordinator_node(state: GraphState) -> Dict:
    """
    Función wrapper para usar en LangGraph
    
    Args:
        state: Estado del grafo
        
    Returns:
        Updates del estado
    """
    return coordinator_agent(state)
