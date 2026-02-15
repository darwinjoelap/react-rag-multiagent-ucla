"""
Prompts OPTIMIZADOS para agentes LangGraph - Reducen tiempos de generación

Cambios vs versión anterior:
- Prompts 70% más cortos
- Ejemplos reducidos de 8 a 3
- Instrucciones más directas
- Sin markdown excesivo
"""

import logging
from app.agents.state import get_conversation_context

logger = logging.getLogger(__name__)

# ==============================================================================
# COORDINADOR - PROMPT ULTRA-COMPACTO
# ==============================================================================

COORDINATOR_SYSTEM_PROMPT = """Coordinador RAG. Analiza y decide.

ACCIONES:
- search: Buscar en documentos (cuando necesites info específica)
- answer: Responder directo (cuando tengas contexto o sea saludo)

FORMATO (OBLIGATORIO):
Thought: [1 línea breve]
Action: [search o answer]
Action Input: [query de búsqueda O respuesta completa]

CONTEXTO: Iteración {iteration}/5 | Docs: {num_docs} | Historial: {has_history}
"""

COORDINATOR_FEW_SHOT_EXAMPLES = """
EJEMPLOS:

Usuario: "¿Qué es machine learning?"
Thought: Concepto técnico, necesito documentos.
Action: search
Action Input: machine learning definición conceptos fundamentales

Usuario: "Hola"
Thought: Saludo simple, no requiere búsqueda.
Action: answer
Action Input: ¡Hola! ¿En qué puedo ayudarte?

Usuario: "Resume lo que encontraste" (con 5 docs en contexto)
Thought: Tengo documentos suficientes, puedo responder.
Action: answer
Action Input: [Basándome en los documentos recuperados...]
"""


def format_coordinator_prompt(state: dict) -> str:
    """Formatear prompt del coordinador - VERSIÓN OPTIMIZADA V2"""
    num_docs = len(state.get("retrieved_documents", []))
    iteration = state.get("iteration", 0)
    
    # Historial (solo si existe)
    history_text = ""
    messages = state.get("messages", [])
    if len(messages) > 1:
        history_text = f"\n\nHISTORIAL:\n{get_conversation_context(state, last_n=3)}\n"
    
    # Query actual
    current_query = state.get("current_query", "")
    
    # Prompt ultra-compacto
    full_prompt = f"""{COORDINATOR_SYSTEM_PROMPT.format(
        iteration=iteration,
        num_docs=num_docs,
        has_history="Sí" if len(messages) > 1 else "No"
    )}

{COORDINATOR_FEW_SHOT_EXAMPLES}
{history_text}

USUARIO: {current_query}

Responde AHORA en formato ReAct (3 líneas):
"""
    
    return full_prompt


# ==============================================================================
# ANSWER NODE - PROMPT ULTRA-COMPACTO
# ==============================================================================

def format_answer_prompt(state: dict) -> str:
    """Formatear prompt del answer - VERSIÓN OPTIMIZADA"""
    
    # Historial (solo últimos 2 turnos)
    history_text = ""
    messages = state.get("messages", [])
    if len(messages) > 1:
        history_text = f"HISTORIAL:\n{get_conversation_context(state, last_n=2)}\n\n"
    
    # Query y documentos
    current_query = state.get("current_query", "")
    docs = state.get("retrieved_documents", [])
    
    # Contexto de documentos (máx 3 docs, 300 chars cada uno)
    context = ""
    if docs:
        context = "DOCUMENTOS:\n"
        for i, doc in enumerate(docs[:3], 1):
            doc_content = doc.get("document", "")[:300]
            source = doc.get("metadata", {}).get("source", "?")
            context += f"[{i}] {source}: {doc_content}...\n\n"
    
    # Prompt ultra-compacto
    prompt = f"""Responde CONCISO (máx 150 palabras).

{history_text}{context}
PREGUNTA: {current_query}

REGLAS:
- Usa historial para referencias ("eso", "aquello")
- Cita fuentes si usas documentos
- Responde SOLO texto, sin formato excesivo

RESPUESTA:
"""
    
    return prompt


# ==============================================================================
# UTILIDADES PARA PARSING
# ==============================================================================

def parse_react_response(response: str) -> dict:
    """
    Parsear respuesta en formato ReAct - VERSIÓN ROBUSTA
    
    Extrae Thought, Action, Action Input incluso si el formato no es perfecto.
    """
    lines = response.strip().split('\n')
    
    result = {
        "thought": "",
        "action": "",
        "action_input": ""
    }
    
    current_field = None
    
    for line in lines:
        line = line.strip()
        
        if line.startswith("Thought:"):
            current_field = "thought"
            result["thought"] = line.replace("Thought:", "").strip()
        elif line.startswith("Action:"):
            current_field = "action"
            result["action"] = line.replace("Action:", "").strip().lower()
        elif line.startswith("Action Input:"):
            current_field = "action_input"
            result["action_input"] = line.replace("Action Input:", "").strip()
        elif current_field and line:
            # Línea de continuación
            result[current_field] += " " + line
    
    # ========== NUEVO: Validaciones y fallbacks ==========
    # Limpiar action_input de posibles artefactos del prompt
    if result["action_input"]:
        # Remover frases del prompt que el LLM pudo copiar
        action_input_clean = result["action_input"]
        
        # Lista de frases a remover
        remove_phrases = [
            "TU RESPUESTA (formato ReAct):",
            "TU RESPUESTA (formato REACT):",
            "Responde AHORA en formato ReAct (3 líneas):",
            "tu respuesta:",
        ]
        
        for phrase in remove_phrases:
            action_input_clean = action_input_clean.replace(phrase, "")
        
        # Limpiar comillas y espacios
        action_input_clean = action_input_clean.strip().strip('"').strip("'").strip()
        
        result["action_input"] = action_input_clean
    
    # Si action_input está vacío pero action es "search", usar el thought como query
    if result["action"] == "search" and not result["action_input"] and result["thought"]:
        result["action_input"] = result["thought"]
        logger.debug(f"⚡ Action input vacío, usando thought como query: '{result['thought'][:50]}...'")
    
    # Si action está vacío o es inválido
    valid_actions = ["search", "answer"]
    if result["action"] not in valid_actions:
        logger.warning(f"⚠️ Acción inválida: '{result['action']}'. Forzando 'search'")
        result["action"] = "search"
        
        # Si no hay action_input, usar toda la respuesta
        if not result["action_input"]:
            result["action_input"] = response[:200]  # Primeros 200 chars
    # =====================================================
    
    return result
