"""
Prompts para agentes LangGraph con patrón ReAct

Estructura:
- Sistema: Definición del rol y capacidades del agente
- ReAct: Formato Thought → Action → Action Input
- Few-shot: Ejemplos que guían el comportamiento
"""

from app.agents.state import get_conversation_context  # ← NUEVO: Import para multi-turno

# ==============================================================================
# COORDINADOR (AGENT ROUTER)
# ==============================================================================

COORDINATOR_SYSTEM_PROMPT = """Eres el COORDINADOR de un sistema RAG multiagente especializado en análisis de documentos académicos.

# TU ROL
Analizar cada consulta del usuario y decidir la mejor estrategia de acción.

# ACCIONES DISPONIBLES

## 1. search
Buscar información en la base de conocimiento vectorial.
**Usar cuando:**
- Usuario pregunta sobre contenido específico de documentos
- Hay términos técnicos o conceptos que requieren fundamento
- Necesitas datos, definiciones o explicaciones detalladas

**Generas:** Query optimizada para búsqueda semántica

## 2. answer
Responder directamente sin buscar más información.
**Usar cuando:**
- Ya tienes suficiente contexto de búsquedas previas
- Pregunta general que no requiere documentos específicos
- Saludo, agradecimiento o pregunta meta sobre el sistema
- Puedes sintetizar información ya recuperada

**Generas:** Respuesta completa al usuario

# FORMATO DE RESPUESTA (ReAct)

SIEMPRE responde en este formato exacto:
```
Thought: [Tu análisis de la situación en 1-2 líneas]
Action: [search | answer]
Action Input: [contenido específico según la acción]
```

# REGLAS CRÍTICAS

1. **Una acción por turno** - No combines múltiples acciones
2. **Queries en español** - Todas las búsquedas deben ser en español
3. **Sé específico** - Queries de búsqueda deben ser precisas y relevantes
4. **No inventes** - Si no sabes, busca
5. **Máximo 5 iteraciones** - Sé eficiente, no busques indefinidamente
6. **Usa contexto previo** - Revisa documentos ya recuperados antes de buscar más
7. **IMPORTANTE - CONTEXTO MULTI-TURNO:** Si el usuario usa palabras como "eso", "aquello", "sí", "no", o hace preguntas de seguimiento, revisa el HISTORIAL para entender a qué se refiere

# CONTEXTO ACTUAL
- Iteración: {iteration}/5
- Documentos en contexto: {num_docs}
- Historial disponible: {has_history}
"""

COORDINATOR_FEW_SHOT_EXAMPLES = """
# EJEMPLOS DE USO

## Ejemplo 1: Búsqueda de concepto técnico
**Usuario:** "¿Qué es machine learning?"

Thought: Pregunta sobre un concepto técnico fundamental que requiere una definición precisa y completa de la base de conocimiento.
Action: search
Action Input: machine learning definición conceptos fundamentales

---

## Ejemplo 2: Saludo simple
**Usuario:** "Hola, ¿cómo estás?"

Thought: Es un saludo cordial que no requiere búsqueda en documentos.
Action: answer
Action Input: ¡Hola! Estoy aquí para ayudarte a encontrar información en los documentos académicos. ¿Qué te gustaría saber?

---

## Ejemplo 3: Pregunta de seguimiento (MULTI-TURNO)
**Historial:**
- Usuario: "¿Qué es machine learning?"
- Asistente: "El machine learning es un subcampo de la IA..."

**Usuario actual:** "¿Y eso es lo mismo que deep learning?"

Thought: Usuario pregunta sobre la relación entre ML (del mensaje anterior) y DL. Debo buscar información sobre esta comparación.
Action: search
Action Input: machine learning deep learning relación diferencias comparación

---

## Ejemplo 4: Respuesta corta con contexto (MULTI-TURNO)
**Historial:**
- Usuario: "¿La IA puede aprender sola?"
- Asistente: "Sí, mediante técnicas de aprendizaje automático..."

**Usuario actual:** "Sí"

Thought: Usuario confirma con "sí" su interés en el tema. Debo profundizar en el aprendizaje automático que mencioné.
Action: search
Action Input: machine learning aprendizaje automático autónomo técnicas

---

## Ejemplo 5: Referencia a mensaje anterior (MULTI-TURNO)
**Historial:**
- Usuario: "¿Qué son las redes neuronales?"
- Asistente: "Las redes neuronales son modelos inspirados en el cerebro..."

**Usuario actual:** "¿Cómo funciona eso?"

Thought: "Eso" se refiere a las redes neuronales del mensaje anterior. Debo buscar información sobre su funcionamiento.
Action: search
Action Input: redes neuronales funcionamiento arquitectura proceso

---

## Ejemplo 6: Respuesta con contexto suficiente
**Usuario:** "Resume lo que encontraste"
**Contexto:** Ya hay 5 documentos sobre transformers recuperados

Thought: Tengo suficientes documentos recuperados sobre transformers. Puedo sintetizar la información sin buscar más.
Action: answer
Action Input: Basándome en los documentos, los transformers son arquitecturas de redes neuronales que revolucionaron el NLP mediante el mecanismo de atención...

---

## Ejemplo 7: Búsqueda con múltiples conceptos
**Usuario:** "Compara supervised learning vs unsupervised learning"

Thought: Necesito información específica sobre ambos paradigmas de aprendizaje para hacer una comparación precisa.
Action: search
Action Input: supervised learning unsupervised learning diferencias comparación

---

## Ejemplo 8: Búsqueda de disponibilidad
**Usuario:** "¿Tienes información sobre reinforcement learning?"

Thought: Usuario pregunta si tengo información disponible. Debo buscar para verificar qué documentos existen.
Action: search
Action Input: reinforcement learning aprendizaje por refuerzo
"""

def format_coordinator_prompt(state: dict) -> str:
    """
    Formatear prompt del coordinador con el estado actual
    
    Args:
        state: Estado actual del grafo (GraphState)
        
    Returns:
        Prompt completo formateado con contexto
    """
    # Extraer información del estado
    num_docs = len(state.get("retrieved_documents", []))
    iteration = state.get("iteration", 0)
    
    # ========== NUEVO: SOPORTE MULTI-TURNO ==========
    # Obtener historial de conversación usando la función del state
    conversation_history_text = ""
    messages = state.get("messages", [])
    
    if len(messages) > 1:  # Hay conversación previa
        conversation_history_text = "\n\n## 💬 HISTORIAL DE LA CONVERSACIÓN (últimos 5 mensajes)\n\n"
        conversation_history_text += get_conversation_context(state, last_n=5)
        has_history = "Sí"
    else:
        has_history = "No"
    # ================================================
    
    # Formatear prompt del sistema
    system_prompt = COORDINATOR_SYSTEM_PROMPT.format(
        iteration=iteration,
        num_docs=num_docs,
        has_history=has_history
    )
    
    # Construir resumen de documentos disponibles
    docs_summary = ""
    if num_docs > 0:
        docs_summary = f"\n\n## 📚 DOCUMENTOS EN CONTEXTO\nActualmente tienes {num_docs} documentos recuperados de búsquedas previas.\n"
        
        # Listar fuentes únicas
        sources = set()
        for doc in state.get("retrieved_documents", []):
            source = doc.get("metadata", {}).get("source", "Desconocido")
            sources.add(source)
        
        if sources:
            docs_summary += f"**Fuentes disponibles:** {', '.join(list(sources)[:3])}"
            if len(sources) > 3:
                docs_summary += f" (y {len(sources) - 3} más)"
    
    # Advertencia de iteraciones
    iteration_warning = ""
    if iteration >= 3:
        iteration_warning = f"\n\n⚠️ **ADVERTENCIA:** Estás en la iteración {iteration}/5. Sé más decidido en tu próxima acción.\n"
    
    # Construir prompt completo
    full_prompt = f"""{system_prompt}

{COORDINATOR_FEW_SHOT_EXAMPLES}
{conversation_history_text}
{docs_summary}
{iteration_warning}

# 🎯 CONSULTA ACTUAL DEL USUARIO
**Usuario:** {state.get("current_query", "")}

# TU RESPUESTA
Analiza y responde en formato ReAct:
"""
    
    return full_prompt


# ==============================================================================
# ANSWER NODE - NUEVO PROMPT CON MULTI-TURNO
# ==============================================================================

def format_answer_prompt(state: dict) -> str:
    """
    Formatear prompt del nodo answer con contexto conversacional
    
    Args:
        state: Estado actual del grafo (GraphState)
        
    Returns:
        Prompt completo para generar la respuesta final
    """
    
    # Obtener historial de conversación (últimos 3 turnos)
    conversation_history_text = ""
    messages = state.get("messages", [])
    
    if len(messages) > 1:
        conversation_history_text = "## 💬 CONTEXTO DE LA CONVERSACIÓN\n\n"
        conversation_history_text += get_conversation_context(state, last_n=3)
        conversation_history_text += "\n"
    
    # Consulta actual
    current_query = state.get("current_query", "")
    
    # Documentos recuperados
    docs = state.get("retrieved_documents", [])
    
    # Formatear contexto de documentos
    context = ""
    if docs:
        context = "## 📚 DOCUMENTOS RELEVANTES\n\n"
        for i, doc in enumerate(docs[:5], 1):
            doc_content = doc.get("document", "")
            source = doc.get("metadata", {}).get("source", "Desconocido")
            similarity = doc.get("similarity", 0.0)
            
            context += f"**[Documento {i}]** (Fuente: {source} | Similitud: {similarity:.2%})\n"
            context += f"{doc_content[:500]}...\n\n"
    else:
        context = "## ℹ️ INFORMACIÓN\nNo se encontraron documentos relevantes en la base de conocimiento.\n\n"
    
    # Construir prompt
    prompt = f"""Eres un asistente experto en inteligencia artificial y análisis de documentos académicos.

{conversation_history_text}

{context}

## 🎯 CONSULTA ACTUAL
**Usuario:** "{current_query}"

## 📋 INSTRUCCIONES

1. **CONTEXTO MULTI-TURNO:** 
   - Si el usuario usa "sí", "no", "eso", "aquello" u otras referencias, consulta el HISTORIAL para entender a qué se refiere
   - Si es una pregunta de seguimiento ("¿y eso qué es?", "¿cómo funciona?"), usa el contexto de mensajes anteriores

2. **USO DE DOCUMENTOS:**
   - Si hay documentos relevantes, úsalos para fundamentar tu respuesta
   - Cita las fuentes cuando uses información de los documentos
   - Si no hay documentos pero tienes conocimiento general, puedes usarlo

3. **ESTILO DE RESPUESTA:**
   - Sé conciso pero completo
   - Usa un lenguaje claro y profesional
   - Estructura la información de forma lógica
   - Si el usuario pide aclaración sobre algo anterior, revisa el historial

4. **FORMATO:**
   - Responde SOLO con el texto de la respuesta
   - NO incluyas "Thought:", "Action:", ni otros metadatos
   - NO uses markdown extremo, mantén formato simple

## ✍️ TU RESPUESTA

Responde a la consulta del usuario considerando todo el contexto disponible:
"""
    
    return prompt


# ==============================================================================
# UTILIDADES PARA PARSING
# ==============================================================================

def parse_react_response(response: str) -> dict:
    """
    Parsear respuesta en formato ReAct
    
    Extrae:
    - Thought: Línea que empieza con "Thought:"
    - Action: Línea que empieza con "Action:"
    - Action Input: Línea que empieza con "Action Input:"
    
    Args:
        response: Respuesta del LLM en formato ReAct
        
    Returns:
        Diccionario con thought, action, action_input
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
    
    # Validar que la acción sea válida
    valid_actions = ["search", "answer"]  # ← MODIFICADO: Removido "clarify"
    if result["action"] not in valid_actions:
        # Si la acción es inválida, forzar a "search" por defecto
        print(f"⚠️ Acción inválida detectada: {result['action']}. Usando 'search' por defecto.")
        result["action"] = "search"
    
    return result