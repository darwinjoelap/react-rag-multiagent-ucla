"""
Answer Node - Genera respuesta final al usuario

Utiliza el contexto recuperado y el LLM para generar
una respuesta fundamentada en los documentos.
"""

import logging
from typing import Dict
from app.agents.state import GraphState, add_trace_step
from app.agents.prompts import format_answer_prompt  # ← NUEVO: Usar prompt con multi-turno
from app.core.llm_config import get_answer_llm  # ← NUEVO: Importar LLM optimizado

logger = logging.getLogger(__name__)


class AnswerNode:
    """
    Nodo que genera la respuesta final al usuario
    """
    
    def __init__(self):
        """Inicializar con LLM optimizado"""
        self.llm = get_answer_llm()  # ← OPTIMIZADO: llama3.2:1b con temp=0.3, tokens=512
        logger.info("AnswerNode inicializado con llama3.2:1b optimizado")
    
    def __call__(self, state: GraphState) -> Dict:
        """
        Generar respuesta final
        
        Args:
            state: Estado actual del grafo
            
        Returns:
            Updates del estado con respuesta final
        """
        logger.info("Answer ejecutando - Generando respuesta")
        
        try:
            # ========== CORREGIDO: Usar current_query en lugar de user_query ==========
            question = state.get("current_query", "")
            
            # Si no hay current_query, intentar con user_query (backward compatibility)
            if not question:
                question = state.get("user_query", "")
            
            # Si aún no hay query, usar action_input como último recurso
            if not question:
                question = state.get("action_input", "")
            # ===========================================================================
            
            action = state.get("action", "answer")
            action_input = state.get("action_input", "")
            
            # Si la acción es "clarify", devolver directamente el action_input
            if action == "clarify":
                final_answer = action_input
                thought = "Solicitando aclaración al usuario"
                logger.info("Respuesta tipo clarify generada")
            
            # ========== FIX: Respuesta directa para queries fuera de dominio ==========
            elif action_input == "out_of_domain":
                final_answer = ("Lo siento, no tengo información sobre ese tema en mi base de conocimiento. "
                               "Mi dominio se limita a temas de Inteligencia Artificial, Machine Learning, "
                               "Redes Neuronales y Agentes Inteligentes. ¿Puedo ayudarte con alguno de estos temas?")
                thought = "Query fuera de dominio - respuesta directa sin documentos"
                logger.info("Respuesta fuera de dominio generada directamente")
            # =========================================================================
            
            # ========== MODIFICADO: Usar format_answer_prompt con multi-turno ==========
            else:
                # Usar el prompt del archivo prompts.py que incluye historial
                logger.debug("Generando respuesta con prompt multi-turno")
                prompt = format_answer_prompt(state)
                
                # Generar respuesta con LLM
                final_answer = self.llm.invoke(prompt).strip()
                
                # Determinar el tipo de respuesta
                docs = state.get("retrieved_documents", [])
                if len(docs) > 0:
                    thought = f"Generando respuesta basada en {len(docs)} documentos recuperados"
                else:
                    thought = "Generando respuesta con conocimiento general (sin documentos)"
                
                logger.info(f"✅ Respuesta generada ({len(final_answer)} caracteres)")
            # ===========================================================================
            
            return {
                "final_answer": final_answer,
                "next_step": "end",
                "should_continue": False,
                **add_trace_step(
                    state,
                    agent="answer",
                    thought=thought,
                    action="answer",
                    observation=f"Respuesta final generada ({len(final_answer)} chars)"
                )
            }
            
        except Exception as e:
            logger.error(f"❌ Error en answer: {str(e)}", exc_info=True)
            
            # ========== MEJORADO: Respuesta de fallback más informativa ==========
            error_message = "Lo siento, ocurrió un error al generar la respuesta."
            
            # Intentar dar una respuesta básica si hay action_input
            action_input = state.get("action_input", "")
            if action_input and len(action_input) > 10:
                error_message = action_input
                logger.info("Usando action_input como respuesta de fallback")
            
            return {
                "final_answer": error_message,
                "error": f"Error en answer: {str(e)}",
                "next_step": "end",
                "should_continue": False,
                **add_trace_step(
                    state,
                    agent="answer",
                    thought="Error al generar respuesta",
                    action="error",
                    observation=str(e)
                )
            }
            # ====================================================================


# Instancia global
answer_node_instance = AnswerNode()


def answer_node(state: GraphState) -> Dict:
    """
    Función wrapper para usar en LangGraph
    
    Args:
        state: Estado del grafo
        
    Returns:
        Updates del estado
    """
    return answer_node_instance(state)
