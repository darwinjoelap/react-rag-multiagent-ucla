"""
Grader Agent - Evalúa relevancia de documentos recuperados

El grader determina si cada documento es relevante para responder
la consulta del usuario. Documentos irrelevantes se filtran.
"""

import logging
from typing import Dict, List
from langchain_ollama import OllamaLLM
from app.agents.state import GraphState, add_trace_step
from app.core.config import settings

logger = logging.getLogger(__name__)


# Prompt para el grader
GRADER_PROMPT = """Eres un evaluador experto que determina si un documento es RELEVANTE para responder una pregunta.

# TU TAREA
Evaluar si el documento contiene información útil para responder la pregunta del usuario.

# CRITERIOS DE RELEVANCIA
Un documento es RELEVANTE si:
- Contiene información directamente relacionada con la pregunta
- Proporciona contexto útil para la respuesta
- Menciona conceptos clave de la pregunta

Un documento NO es relevante si:
- Habla de un tema completamente diferente
- Solo menciona palabras clave sin contexto útil
- Es demasiado genérico o vago

# FORMATO DE RESPUESTA
Responde SOLO con una palabra:
- "relevante" si el documento es útil
- "irrelevante" si el documento no ayuda

# PREGUNTA DEL USUARIO
{question}

# DOCUMENTO A EVALUAR
{document}

# TU EVALUACIÓN
"""


class GraderAgent:
    """
    Agente que evalúa la relevancia de documentos recuperados
    
    Para cada documento, determina si es útil para responder
    la consulta del usuario.
    """
    
    def __init__(self):
        """Inicializar grader con LLM"""
        self.llm = OllamaLLM(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_MODEL,
            temperature=0.0  # Temperatura 0 para evaluaciones binarias consistentes
        )
        logger.info("GraderAgent inicializado")
    
    def grade_document(self, question: str, document: str) -> bool:
        """
        Evaluar un documento individual
        
        Args:
            question: Pregunta del usuario
            document: Contenido del documento
            
        Returns:
            True si es relevante, False si no
        """
        # Formatear prompt
        prompt = GRADER_PROMPT.format(
            question=question,
            document=document[:1000]  # Limitar a 1000 chars para eficiencia
        )
        
        # Obtener evaluación
        response = self.llm.invoke(prompt).strip().lower()
        
        # Parsear respuesta
        #is_relevant = "relevante" in response and "irrelevante" not in response
        is_relevant = "relevante" in response or len(response) < 20
        
        logger.debug(f"Documento evaluado: {'✅ relevante' if is_relevant else '❌ irrelevante'}")
        
        return is_relevant
    
    def __call__(self, state: GraphState) -> Dict:
        """
        Ejecutar nodo grader - Evaluar todos los documentos recuperados
        
        Args:
            state: Estado actual del grafo
            
        Returns:
            Updates del estado con documentos filtrados
        """
        logger.info("Grader ejecutando - Evaluando documentos")
        
        try:
            question = state["user_query"]
            documents = state.get("retrieved_documents", [])
            
            if not documents:
                logger.warning("No hay documentos para evaluar")
                return {
                    "observation": "No hay documentos para evaluar",
                    **add_trace_step(
                        state,
                        agent="grader",
                        thought="Sin documentos para evaluar",
                        action="skip",
                        observation="No documents to grade"
                    )
                }
            
            # Evaluar cada documento
            relevant_docs = []
            irrelevant_count = 0
            
            for doc in documents:
                doc_content = doc.get("document", "")
                is_relevant = self.grade_document(question, doc_content)
                
                if is_relevant:
                    relevant_docs.append(doc)
                else:
                    irrelevant_count += 1
            
            logger.info(f"Documentos: {len(relevant_docs)} relevantes, {irrelevant_count} irrelevantes")
            
            # Construir observación
            observation = f"Evaluados {len(documents)} documentos: {len(relevant_docs)} relevantes, {irrelevant_count} irrelevantes"
            
            # Determinar siguiente paso
            if len(relevant_docs) == 0:
                # Sin documentos relevantes - reescribir query
                next_step = "rewrite"
                thought = "No hay documentos relevantes, necesito reescribir la consulta"
            else:
                # Hay documentos relevantes - generar respuesta
                next_step = "answer"
                thought = f"Encontrados {len(relevant_docs)} documentos relevantes, proceder a responder"
            
            return {
                "retrieved_documents": relevant_docs,  # Sobrescribir con solo relevantes
                "observation": observation,
                "next_step": next_step,
                **add_trace_step(
                    state,
                    agent="grader",
                    thought=thought,
                    action="grade",
                    observation=observation
                )
            }
            
        except Exception as e:
            logger.error(f"Error en grader: {str(e)}")
            return {
                "error": f"Error en grader: {str(e)}",
                "next_step": "answer",  # Continuar a answer aunque haya error
                **add_trace_step(
                    state,
                    agent="grader",
                    thought="Error al evaluar documentos",
                    action="error",
                    observation=str(e)
                )
            }


# Instancia global
grader_agent = GraderAgent()


def grader_node(state: GraphState) -> Dict:
    """
    Función wrapper para usar en LangGraph
    
    Args:
        state: Estado del grafo
        
    Returns:
        Updates del estado
    """
    return grader_agent(state)