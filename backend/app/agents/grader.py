"""
Grader Agent - Evalúa relevancia de documentos recuperados

El grader determina si cada documento es relevante para responder
la consulta del usuario. Documentos irrelevantes se filtran.
"""

import logging
from typing import Dict, List
from app.agents.state import GraphState, add_trace_step
from app.core.llm_config import get_grader_llm  # ← NUEVO: Importar LLM optimizado

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
        """Inicializar grader con LLM optimizado"""
        self.llm = get_grader_llm()  # ← OPTIMIZADO: llama3.2:1b con temp=0.0, tokens=128
        logger.info("GraderAgent inicializado con llama3.2:1b optimizado")
    
    def grade_document(self, question: str, document: str, similarity: float = 0.0) -> bool:
        """
        Evaluar un documento - VERSIÓN OPTIMIZADA SIN LLM
        
        Usa solo el similarity score para determinar relevancia.
        Esto reduce el tiempo de 38s a <1s.
        
        Args:
            question: Pregunta del usuario
            document: Contenido del documento
            similarity: Score de similitud del vector store
            
        Returns:
            True si es relevante, False si no
        """
        # Estrategia simple: usar solo similarity score
        # Threshold: 0.25 (ajustable según necesidad)
        threshold = 0.25
        is_relevant = similarity >= threshold
        
        logger.debug(f"Doc evaluado: sim={similarity:.4f} → {'✅ relevante' if is_relevant else '❌ irrelevante'}")
        
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
            # ========== CORREGIDO: Usar current_query ==========
            question = state.get("current_query", "")
            
            # Fallback si no hay current_query
            if not question:
                question = state.get("user_query", "")
            if not question:
                question = state.get("action_input", "")
            # ===================================================
            
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
            
            # 🆕 LOGGING DETALLADO POR DOCUMENTO
            logger.info(f"📊 Evaluando {len(documents)} documentos:")
            
            # Evaluar cada documento usando similarity score
            relevant_docs = []
            irrelevant_docs = []
            
            for i, doc in enumerate(documents, 1):
                doc_content = doc.get("document", "")
                similarity = doc.get("similarity", 0.0)
                
                # Extraer metadata
                source = doc.get('metadata', {}).get('source', 'unknown')
                filename = source.split('/')[-1] if '/' in source else source
                chunk_id = doc.get('metadata', {}).get('chunk_id', 'N/A')
                
                # Evaluar relevancia
                is_relevant = self.grade_document(question, doc_content, similarity)
                
                # 🆕 LOG INDIVIDUAL POR DOCUMENTO
                if is_relevant:
                    relevant_docs.append(doc)
                    logger.info(f"  ✅ [{i}] {filename} (chunk {chunk_id}, sim={similarity:.4f}) - RELEVANTE")
                else:
                    irrelevant_docs.append(doc)
                    logger.info(f"  ❌ [{i}] {filename} (chunk {chunk_id}, sim={similarity:.4f}) - IRRELEVANTE")
            
            logger.info(f"Documentos: {len(relevant_docs)} relevantes, {len(irrelevant_docs)} irrelevantes")
            
            # Construir observación
            observation = f"Evaluados {len(documents)} documentos: {len(relevant_docs)} relevantes, {len(irrelevant_docs)} irrelevantes"
            
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
