"""
Prompts para agentes LangGraph con patrón ReAct

Estructura:
- Sistema: Definición del rol y capacidades del agente
- ReAct: Formato Thought → Action → Action Input
- Few-shot: Ejemplos que guían el comportamiento
"""

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

## 3. clarify
Solicitar más información al usuario.
**Usar cuando:**
- Consulta ambigua o muy vaga
- Falta contexto esencial para responder bien
- Múltiples interpretaciones posibles

**Generas:** Pregunta específica para obtener claridad

# FORMATO DE RESPUESTA (ReAct)

SIEMPRE responde en este formato exacto:
```
Thought: [Tu análisis de la situación en 1-2 líneas]
Action: [search | answer | clarify]
Action Input: [contenido específico según la acción]
```

# REGLAS CRÍTICAS

1. **Una acción por turno** - No combines múltiples acciones
2. **Queries en español** - Todas las búsquedas deben ser en español
3. **Sé específico** - Queries de búsqueda deben ser precisas y relevantes
4. **No inventes** - Si no sabes, busca o pide aclaración
5. **Máximo 5 iteraciones** - Sé eficiente, no busques indefinidamente
6. **Usa contexto previo** - Revisa documentos ya recuperados antes de buscar más

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

## Ejemplo 3: Consulta vaga
**Usuario:** "Explícame eso"

Thought: La consulta es demasiado vaga - no sé a qué se refiere "eso". Necesito contexto.
Action: clarify
Action Input: ¿A qué tema específico te refieres? ¿Podrías darme más detalles sobre lo que deseas que explique?

---

## Ejemplo 4: Pregunta de seguimiento
**Usuario:** "¿Y cómo se relaciona con deep learning?"
**Contexto:** Conversación previa sobre machine learning

Thought: Usuario hace seguimiento sobre la relación ML-DL. Debo buscar información específica sobre esta conexión.
Action: search
Action Input: machine learning deep learning relación diferencias conexión

---

## Ejemplo 5: Respuesta con contexto suficiente
**Usuario:** "Resume lo que encontraste"
**Contexto:** Ya hay 5 documentos sobre transformers recuperados

Thought: Tengo suficientes documentos recuperados sobre transformers. Puedo sintetizar la información sin buscar más.
Action: answer
Action Input: Basándome en los documentos, los transformers son arquitecturas de redes neuronales que revolucionaron el NLP mediante el mecanismo de atención...

---

## Ejemplo 6: Búsqueda con múltiples conceptos
**Usuario:** "Compara supervised learning vs unsupervised learning"

Thought: Necesito información específica sobre ambos paradigmas de aprendizaje para hacer una comparación precisa.
Action: search
Action Input: supervised learning unsupervised learning diferencias comparación

---

## Ejemplo 7: Pregunta sobre disponibilidad
**Usuario:** "¿Tienes información sobre reinforcement learning?"

Thought: Usuario pregunta si tengo información disponible. Debo buscar para verificar qué documentos existen.
Action: search
Action Input: reinforcement learning aprendizaje por refuerzo

---

## Ejemplo 8: Aclaración de alcance
**Usuario:** "Dame todo sobre IA"

Thought: Petición demasiado amplia. "Todo sobre IA" es excesivamente general. Debo pedir especificidad.
Action: clarify
Action Input: La inteligencia artificial es un campo muy amplio. ¿Qué aspecto específico te interesa? Por ejemplo: machine learning, visión computacional, procesamiento de lenguaje natural, aplicaciones, historia, etc.
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
    has_history = "Sí" if len(state.get("conversation_history", [])) > 1 else "No"
    iteration = state.get("iteration", 0)
    
    # Formatear prompt del sistema
    system_prompt = COORDINATOR_SYSTEM_PROMPT.format(
        iteration=iteration,
        num_docs=num_docs,
        has_history=has_history
    )
    
    # Construir contexto de conversación reciente
    conversation_context = ""
    history = state.get("conversation_history", [])
    if len(history) > 1:  # Más que solo la consulta actual
        conversation_context = "\n\n## HISTORIAL RECIENTE\n"
        for msg in history[-4:]:  # Últimos 4 mensajes
            role = "USUARIO" if msg["role"] == "user" else "ASISTENTE"
            content = msg["content"]
            conversation_context += f"**{role}:** {content}\n"
    
    # Construir resumen de documentos disponibles
    docs_summary = ""
    if num_docs > 0:
        docs_summary = f"\n\n## DOCUMENTOS EN CONTEXTO\nActualmente tienes {num_docs} documentos recuperados de búsquedas previas.\n"
        
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
{conversation_context}
{docs_summary}
{iteration_warning}

# CONSULTA ACTUAL DEL USUARIO
**Usuario:** {state.get("user_query", "")}

# TU RESPUESTA
Analiza y responde en formato ReAct:
"""
    
    return full_prompt


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
    valid_actions = ["search", "answer", "clarify"]
    if result["action"] not in valid_actions:
        raise ValueError(f"Acción inválida: {result['action']}. Debe ser una de: {valid_actions}")
    
    return result