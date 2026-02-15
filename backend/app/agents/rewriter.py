"""
Rewriter Agent - Reescribe queries para mejorar resultados de búsqueda

Cuando la búsqueda inicial no obtiene documentos relevantes,
el rewriter reformula la consulta para mejorar las probabilidades
de encontrar información útil.
"""

import logging
from typing import Dict
from app.agents.state import GraphState, add_trace_step, increment_retry  # ← AGREGADO increment_retry
from app.core.llm_config import get_rewriter_llm  # ← NUEVO: Importar LLM optimizado

logger = logging.getLogger(__name__)


# Prompt para el rewriter
REWRITER_PROMPT = """Eres un experto en reformular consultas para mejorar resultados de búsqueda semántica.

# TU TAREA
Reescribir la pregunta original para obtener mejores resultados al buscar en una base de documentos académicos.

# ESTRATEGIAS DE REESCRITURA
1. **Expandir conceptos**: Agregar sinónimos o términos relacionados
2. **Simplificar**: Remover ambigüedad o generalización excesiva
3. **Especificar**: Agregar contexto técnico relevante
4. **Traducir a keywords**: Convertir preguntas en términos de búsqueda

# EJEMPLO
Original: "¿Cómo funciona eso?"
Reescrita: "algoritmos machine learning funcionamiento principios básicos"

Original: "Explícame redes neuronales"
Reescrita: "redes neuronales arquitectura capas neuronas aprendizaje profundo"

Original: "IA en medicina"
Reescrita: "inteligencia artificial aplicaciones médicas diagnóstico tratamiento"

# REGLAS
- Mantén el idioma español
- Prioriza términos técnicos y específicos
- Elimina palabras de relleno ("por favor", "gracias", etc.)
- Convierte preguntas en afirmaciones o keywords
- Máximo 10-15 palabras

# CONSULTA ORIGINAL
{original_query}

# CONSULTA ANTERIOR (que no funcionó)
{previous_query}

# TU CONSULTA REESCRITA
Escribe SOLO la nueva consulta, sin explicaciones:
"""


class RewriterAgent:
    """
    Agente que reescribe queries para mejorar búsquedas
    
    Utilizado cuando los documentos recuperados no son relevantes
    o cuando la búsqueda inicial no da buenos resultados.
    """
    
    def __init__(self):
        """Inicializar rewriter con LLM optimizado"""
        self.llm = get_rewriter_llm()  # ← OPTIMIZADO: llama3.2:1b con temp=0.5, tokens=256
        logger.info("RewriterAgent inicializado con llama3.2:1b optimizado")
    
    def rewrite_query(self, original_query: str, previous_query: str = None) -> str:
        """
        Reescribir una consulta
        
        Args:
            original_query: Pregunta original del usuario
            previous_query: Query anterior que no funcionó (opcional)
            
        Returns:
            Query reescrita optimizada para búsqueda
        """
        if previous_query is None:
            previous_query = original_query
        
        # Formatear prompt
        prompt = REWRITER_PROMPT.format(
            original_query=original_query,
            previous_query=previous_query
        )
        
        # Obtener reescritura
        rewritten = self.llm.invoke(prompt).strip()
        
        # Limpiar respuesta (remover comillas, puntos finales, etc.)
        rewritten = rewritten.strip('"').strip("'").strip('.').strip()
        
        logger.info(f"Query reescrita: '{previous_query}' → '{rewritten}'")
        
        return rewritten
    
    def __call__(self, state: GraphState) -> Dict:
        """
        Ejecutar nodo rewriter - Reescribir query y volver a buscar
        
        Args:
            state: Estado actual del grafo
            
        Returns:
            Updates del estado con nueva query
        """
        # ========== MODIFICADO: Log del retry_count actual ==========
        current_retry = state.get("retry_count", 0)
        logger.info(f"Rewriter ejecutando - Retry actual: {current_retry}")
        # =============================================================
        
        try:
            original_query = state.get("user_query", "")
            previous_action_input = state.get("action_input", original_query)
            
            # ========== MODIFICADO: Verificar límite antes de reescribir ==========
            if current_retry >= 2:
                logger.warning(f"⚠️ Rewriter: Límite de reintentos alcanzado ({current_retry}). Forzando respuesta.")
                return {
                    "action": "answer",
                    "action_input": f"No encontré documentos específicos, pero puedo responder sobre: {original_query}",
                    "next_step": "answer",
                    "observation": "Límite de reintentos alcanzado",
                    **add_trace_step(
                        state,
                        agent="rewriter",
                        thought="Límite de reintentos alcanzado",
                        action="answer",
                        observation="Forzando respuesta final"
                    )
                }
            # =======================================================================
            
            # Reescribir query
            new_query = self.rewrite_query(original_query, previous_action_input)
            
            thought = f"Query anterior no dio resultados relevantes. Reescribiendo: '{previous_action_input}' → '{new_query}'"
            
            # ========== MODIFICADO: Usar increment_retry() ==========
            return {
                "action": "search",
                "action_input": new_query,
                "next_step": "search",
                "observation": f"Query reescrita: {new_query}",
                **increment_retry(state),  # ← USAR FUNCIÓN DEL STATE
                **add_trace_step(
                    state,
                    agent="rewriter",
                    thought=thought,
                    action="rewrite",
                    observation=f"Nueva query: {new_query} | Retry: {current_retry + 1}"
                )
            }
            # =========================================================
            
        except Exception as e:
            logger.error(f"Error en rewriter: {str(e)}")
            # Si falla, continuar con respuesta usando lo que hay
            return {
                "error": f"Error en rewriter: {str(e)}",
                "action": "answer",
                "next_step": "answer",
                **add_trace_step(
                    state,
                    agent="rewriter",
                    thought="Error al reescribir query",
                    action="error",
                    observation=str(e)
                )
            }


# Instancia global
rewriter_agent = RewriterAgent()


def rewriter_node(state: GraphState) -> Dict:
    """
    Función wrapper para usar en LangGraph
    
    Args:
        state: Estado del grafo
        
    Returns:
        Updates del estado
    """
    return rewriter_agent(state)
