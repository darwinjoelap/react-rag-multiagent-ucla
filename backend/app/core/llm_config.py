"""
Configuración centralizada del LLM para todos los agentes

Este módulo proporciona una configuración optimizada de Ollama
para maximizar la velocidad de respuesta usando llama3.2:1b
"""

import logging
from langchain_ollama import OllamaLLM
from app.core.config import settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.1, num_predict: int = 512) -> OllamaLLM:
    """
    Crear instancia de LLM optimizada para velocidad
    
    Configuración optimizada para llama3.2:1b que reduce significativamente
    los tiempos de respuesta (de 2-3 min a 30-60 segundos)
    
    Args:
        temperature: Temperatura para generación
            - 0.0 = Completamente determinista (mejor para clasificación)
            - 0.1-0.3 = Ligeramente creativo (bueno para coordinación)
            - 0.5-0.7 = Moderadamente creativo (bueno para respuestas)
        num_predict: Máximo de tokens a generar
            - 256 = Respuestas muy cortas (coordinador, grader)
            - 512 = Respuestas moderadas (answer general)
            - 1024 = Respuestas largas (solo si es necesario)
    
    Returns:
        Instancia configurada de OllamaLLM
    """
    try:
        llm = OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model="llama3.2:1b",  # Modelo optimizado para velocidad
            temperature=temperature,
            num_predict=num_predict,  # Limitar tokens de salida
            num_ctx=2048,             # Context window reducido (de 4096)
            repeat_penalty=1.1,       # Evitar repeticiones
        )
        
        logger.debug(f"LLM creado: llama3.2:1b (temp={temperature}, max_tokens={num_predict})")
        return llm
        
    except Exception as e:
        logger.error(f"Error al crear LLM: {str(e)}")
        raise


def get_coordinator_llm() -> OllamaLLM:
    """
    LLM optimizado para el Coordinador
    
    - Temperatura baja para decisiones consistentes
    - Tokens limitados ya que solo necesita generar Thought/Action/ActionInput
    """
    return get_llm(temperature=0.0, num_predict=256)


def get_grader_llm() -> OllamaLLM:
    """
    LLM optimizado para el Grader
    
    - Temperatura 0 para evaluaciones binarias consistentes
    - Tokens mínimos ya que solo responde "relevante" o "irrelevante"
    """
    return get_llm(temperature=0.0, num_predict=128)


def get_rewriter_llm() -> OllamaLLM:
    """
    LLM optimizado para el Rewriter
    
    - Temperatura media para creatividad controlada
    - Tokens limitados ya que solo reescribe queries cortas
    """
    return get_llm(temperature=0.5, num_predict=256)


def get_answer_llm() -> OllamaLLM:
    """
    LLM optimizado para el Answer
    
    - Temperatura moderada para respuestas naturales
    - Más tokens para respuestas completas pero concisas
    """
    return get_llm(temperature=0.3, num_predict=512)
