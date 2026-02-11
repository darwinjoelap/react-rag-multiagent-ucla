"""
Servicio de LLM usando Ollama
"""
from langchain_ollama import OllamaLLM
import logging

logger = logging.getLogger(__name__)


def get_llm(model: str = "llama3.2:latest", temperature: float = 0):
    """
    Obtener instancia del LLM
    
    Args:
        model: Nombre del modelo de Ollama
        temperature: Temperatura para generación (0 = determinista)
        
    Returns:
        Instancia de OllamaLLM
    """
    try:
        llm = OllamaLLM(
            model=model,
            temperature=temperature,
            base_url="http://localhost:11434"
        )
        logger.info(f"LLM inicializado: {model}")
        return llm
    except Exception as e:
        logger.error(f"Error inicializando LLM: {e}")
        raise