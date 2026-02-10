"""
Answer Node - Genera respuesta final al usuario

Utiliza el contexto recuperado y el LLM para generar
una respuesta fundamentada en los documentos.
"""

import logging
from typing import Dict
from langchain_ollama import OllamaLLM
from app.agents.state import GraphState, add_trace_step
from app.core.config import settings

logger = logging.getLogger(__name__)


# Prompt para generar respuesta
ANSWER_PROMPT = """Eres un asistente académico experto que responde preguntas basándose en documentos proporcionados.

# TU TAREA
Responder la pregunta del usuario utilizando ÚNICAMENTE la información de los documentos proporcionados.

# REGLAS IMPORTANTES
1. **Usa solo los documentos**: No inventes información
2. **Sé preciso**: Responde directamente la pregunta
3. **Cita fuentes**: Menciona de qué documento viene la información
4. **Admite limitaciones**: Si los documentos no tienen la respuesta, dilo claramente
5. **Sé conciso**: Respuesta clara y al punto

# DOCUMENTOS DISPONIBLES
{context}

# PREGUNTA DEL USUARIO
{question}

# TU RESPUESTA
Basándote en los documentos anteriores, responde de forma clara y precisa:
"""


class AnswerNode:
    """
    Nodo que genera la respuesta final al usuario
    """
    
    def __init__(self):
        """Inicializar con LLM"""
        self.llm = OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.7  # Temperatura moderada para respuestas naturales
        )
        logger.info("AnswerNode inicializado")
    
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
            question = state["user_query"]
            action = state.get("action", "answer")
            action_input = state.get("action_input", "")
            
            # Si la acción es "clarify", devolver directamente el action_input
            if action == "clarify":
                final_answer = action_input
                thought = "Solicitando aclaración al usuario"
            
            # Si hay contexto, generar respuesta basada en documentos
            elif state.get("relevant_context"):
                context = state["relevant_context"]
                
                # Formatear prompt
                prompt = ANSWER_PROMPT.format(
                    context=context[:3000],  # Limitar contexto
                    question=question
                )
                
                # Generar respuesta
                logger.debug("Generando respuesta con LLM")
                final_answer = self.llm.invoke(prompt)
                thought = "Generando respuesta basada en documentos recuperados"
            
            # Si no hay contexto pero hay action_input (respuesta directa del coordinador)
            elif action_input:
                final_answer = action_input
                thought = "Respuesta directa sin búsqueda de documentos"
            
            # Caso por defecto
            else:
                final_answer = "Lo siento, no pude encontrar información relevante para responder tu pregunta. ¿Podrías reformularla o proporcionar más detalles?"
                thought = "Sin contexto ni respuesta disponible"
            
            logger.info("Respuesta generada exitosamente")
            
            return {
                "final_answer": final_answer,
                "next_step": "end",
                "should_continue": False,
                **add_trace_step(
                    state,
                    agent="answer",
                    thought=thought,
                    action="answer",
                    observation="Respuesta final generada"
                )
            }
            
        except Exception as e:
            logger.error(f"Error en answer: {str(e)}")
            return {
                "final_answer": f"Lo siento, ocurrió un error al generar la respuesta: {str(e)}",
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