# Backend Architecture - RAG Multiagent System

## 📋 Overview
Sistema RAG (Retrieval Augmented Generation) multiagente usando LangGraph con patrón ReAct para responder consultas sobre documentos académicos de la UCLA.

**Tecnologías principales:**
- **LangGraph**: Orquestación de agentes
- **Ollama (Llama 3.2)**: Modelo de lenguaje local
- **ChromaDB**: Base de datos vectorial
- **LangChain**: Framework de integración
- **Sentence Transformers**: Embeddings semánticos

---

## 🏗️ Arquitectura General
```
┌─────────────────────────────────────────────────────────────┐
│                     Usuario                                  │
└──────────────────────┬──────────────────────────────────────┘
                       │ Query
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  COORDINATOR AGENT                           │
│  (Analiza query y decide estrategia: search/answer/clarify) │
└──────┬──────────────────────────────────────────────────────┘
       │
       ├──────► search ──────┐
       │                      │
       ├──────► answer ───────┼──────► RESPUESTA FINAL
       │                      │
       └──────► clarify ──────┘
                              │
                              ▼
                     ┌────────────────┐
                     │  SEARCH NODE   │
                     │  (ChromaDB)    │
                     └────────┬───────┘
                              │ Documentos
                              ▼
                     ┌────────────────┐
                     │  GRADER AGENT  │
                     │  (Evalúa docs) │
                     └────────┬───────┘
                              │
                ┌─────────────┴─────────────┐
                │                           │
           Relevantes                   Irrelevantes
                │                           │
                ▼                           ▼
         ┌─────────────┐           ┌──────────────┐
         │ ANSWER NODE │           │ REWRITER NODE│
         └─────────────┘           └──────┬───────┘
                                          │
                                          └──► SEARCH (retry)
```

---

## 🧩 Componentes Principales

### 1. **State Management** (`state.py`)

**GraphState**: Estado compartido entre todos los nodos del grafo.
```python
class GraphState(TypedDict):
    # Query del usuario
    user_query: str
    
    # Patrón ReAct
    thought: str          # Razonamiento del agente
    action: str           # Acción a ejecutar (search/answer/clarify)
    action_input: str     # Input para la acción
    observation: str      # Resultado de la acción
    
    # Documentos
    retrieved_documents: List[Dict]
    relevant_context: str
    
    # Respuesta
    final_answer: str
    
    # Control de flujo
    next_step: str
    should_continue: bool
    iteration: int
    
    # Historia
    conversation_history: List[Dict]
    trace: List[Dict]     # Traza ReAct completa
    
    # Metadata
    timestamp: str
```

**Funciones clave:**
- `create_initial_state(query)`: Inicializa estado para nueva consulta
- Gestiona hasta **5 iteraciones** para prevenir bucles infinitos

---

### 2. **Coordinator Agent** (`coordinator.py`)

**Rol**: Cerebro del sistema. Analiza cada query y decide la mejor estrategia.

**Acciones disponibles:**

| Acción    | Cuándo usar                                          | Output                    |
|-----------|------------------------------------------------------|---------------------------|
| `search`  | Query técnica que requiere buscar en documentos     | Query optimizada para búsqueda |
| `answer`  | Ya hay suficiente contexto o es pregunta general    | Respuesta directa          |
| `clarify` | Query ambigua, necesita más información             | Pregunta de clarificación  |

**Proceso:**
1. Recibe query + contexto (historial, docs previos)
2. Genera prompt con formato ReAct
3. LLM (Llama 3.2) razona y decide acción
4. Parsea respuesta: `Thought`, `Action`, `Action Input`
5. Actualiza estado y traza

**Ejemplo de output:**
```
Thought: Pregunta técnica sobre un concepto de IA que requiere búsqueda en la base de conocimiento.
Action: search
Action Input: inteligencia artificial definición conceptos fundamentales
```

---

### 3. **Search Node** (`search_node.py`)

**Rol**: Recuperar documentos relevantes de ChromaDB.

**Proceso:**
1. Recibe `action_input` como query de búsqueda
2. Usa `RetrieverService` con parámetros:
   - `top_k = 5`: Máximo 5 documentos
   - `similarity_threshold = 0.2`: Umbral de similitud coseno
3. Retorna documentos con scores de similitud
4. Actualiza `retrieved_documents` en el estado
5. Next step: `grader`

**Logging:**
```python
🔍 Search query: 'inteligencia artificial definición'
📊 Resultados encontrados: 5
  ✅ Doc: sim=0.4321
  ✅ Doc: sim=0.3307
  ...
```

---

### 4. **Grader Agent** (`grader.py`)

**Rol**: Evaluar si los documentos recuperados son relevantes para la query.

**Proceso:**
1. Para cada documento recuperado:
   - Genera prompt: `¿Este documento es relevante para "{query}"?`
   - LLM responde: `relevant` o `irrelevant`
   - Filtra documentos irrelevantes

2. Decisión de routing:
```python
   if documentos_relevantes > 0:
       next_step = "answer"
   else:
       next_step = "rewrite"
```

**Ejemplo:**
```
Input: 3 documentos sobre IA
Query: "¿Qué es machine learning?"

Evaluación:
  Doc 1: "La IA es..." → relevant ✅
  Doc 2: "El clima es..." → irrelevant ❌
  Doc 3: "ML es una rama de la IA..." → relevant ✅

Output: 2 documentos relevantes → next_step = "answer"
```

---

### 5. **Rewriter Agent** (`rewriter.py`)

**Rol**: Reescribir queries que no produjeron resultados relevantes.

**Cuándo se activa:**
- El Grader determinó que no hay documentos relevantes
- La búsqueda inicial fue muy vaga o mal formulada

**Proceso:**
1. Recibe:
   - Query original del usuario
   - Query anterior que falló
2. LLM genera versión mejorada:
   - Más específica
   - Términos técnicos apropiados
   - Enfoque diferente
3. Retorna a `search` con nueva query

**Ejemplo:**
```
Query original: "Explícame eso"
Query anterior: "eso explicación"

Reescritura: "algoritmos de aprendizaje supervisado en machine learning"
```

---

### 6. **Answer Node** (`answer_node.py`)

**Rol**: Generar la respuesta final al usuario.

**Modos de operación:**

#### A) **Con contexto de documentos**
```python
Input:
  - retrieved_documents (filtrados por Grader)
  - user_query

Proceso:
  1. Formatea contexto de documentos:
     [Documento 1 - fuente.pdf]
     Contenido...
     
     [Documento 2 - fuente2.pdf]
     Contenido...
  
  2. Genera prompt para LLM:
     "Basándote en estos documentos, responde: {query}"
  
  3. LLM sintetiza información
  
  4. Retorna respuesta citando fuentes

Output:
  "Según Russell-Norvig.pdf, la inteligencia artificial es..."
```

#### B) **Respuesta directa (sin búsqueda)**
```python
# Para saludos, preguntas meta, etc.
User: "Hola, ¿cómo estás?"
Answer: "¡Hola! Estoy aquí para ayudarte..."
```

#### C) **Clarificación**
```python
User: "Explícame eso"
Answer: "¿A qué tema específico te refieres? ¿Podrías darme más detalles?"
```

**Control de flujo:**
```python
should_continue = False  # Siempre termina el grafo
next_step = END
```

---

### 7. **Graph Orchestration** (`graph.py`)

**Estructura LangGraph:**
```python
workflow = StateGraph(GraphState)

# Nodos
workflow.add_node("coordinator", coordinator_node)
workflow.add_node("search", search_node)
workflow.add_node("grader", grader_node)
workflow.add_node("rewrite", rewriter_node)
workflow.add_node("answer", answer_node)

# Edges
workflow.set_entry_point("coordinator")

workflow.add_conditional_edges(
    "coordinator",
    route_decision,  # search / answer / END
)

workflow.add_edge("search", "grader")

workflow.add_conditional_edges(
    "grader",
    route_grader,  # answer / rewrite
)

workflow.add_edge("rewrite", "search")
workflow.add_edge("answer", END)
```

**Funciones de routing:**
```python
def route_decision(state: GraphState) -> str:
    """Enruta desde Coordinator"""
    action = state.get("action", "answer")
    
    if action == "search":
        return "search"
    else:  # answer o clarify
        return "answer"

def route_grader(state: GraphState) -> str:
    """Enruta desde Grader"""
    next_step = state.get("next_step", "answer")
    
    if next_step == "rewrite":
        return "rewrite"
    else:
        return "answer"
```

**Función principal:**
```python
def run_graph(user_query: str) -> GraphState:
    """
    Ejecuta el grafo completo
    
    Args:
        user_query: Pregunta del usuario
        
    Returns:
        GraphState con final_answer y traza completa
    """
    initial_state = create_initial_state(user_query)
    graph = get_graph()
    final_state = graph.invoke(initial_state)
    return final_state
```

---

## 🔄 Flujos de Ejemplo

### **Flujo 1: Query Técnica Exitosa**
```
Usuario: "¿Qué es inteligencia artificial?"
    ↓
Coordinator:
    Thought: "Pregunta técnica que requiere documentos"
    Action: search
    Action Input: "inteligencia artificial definición conceptos"
    ↓
Search:
    → Recupera 5 documentos de ChromaDB
    → Similarities: [0.85, 0.78, 0.65, 0.52, 0.45]
    ↓
Grader:
    → Evalúa cada documento
    → 4 relevantes, 1 irrelevante
    → Next: answer
    ↓
Answer:
    → Formatea contexto de 4 docs
    → LLM genera síntesis
    → "Según Russell-Norvig.pdf, la IA es el campo de estudio..."
    ↓
FIN ✅
```

---

### **Flujo 2: Query Vaga → Rewrite → Éxito**
```
Usuario: "Explícame eso"
    ↓
Coordinator:
    Thought: "Query demasiado vaga"
    Action: clarify
    Action Input: "¿A qué tema te refieres?"
    ↓
Answer:
    → "¿A qué tema específico te refieres? ¿Podrías darme más detalles?"
    ↓
FIN ✅
```

---

### **Flujo 3: Búsqueda Fallida → Rewrite**
```
Usuario: "Dame info sobre XYZ raro"
    ↓
Coordinator:
    Action: search
    Action Input: "XYZ raro"
    ↓
Search:
    → Recupera 3 documentos
    ↓
Grader:
    → Los 3 son irrelevantes
    → Next: rewrite
    ↓
Rewriter:
    → "tecnología XYZ sistemas computacionales"
    → Action: search
    ↓
Search (2do intento):
    → Recupera documentos diferentes
    ↓
Grader:
    → 2 relevantes
    → Next: answer
    ↓
Answer:
    → Genera respuesta con docs relevantes
    ↓
FIN ✅
```

---

## ⚙️ Configuración

### **LLM Settings** (`app/services/llm.py`)
```python
model = "llama3.2:latest"
temperature = 0  # Determinista
base_url = "http://localhost:11434"
```

### **Retriever Settings** (`app/services/retriever.py`)
```python
top_k = 5
similarity_threshold = 0.2  # Cosine similarity
```

### **Embeddings** (`app/services/embeddings.py`)
```python
model_name = "sentence-transformers/all-MiniLM-L6-v2"
dimension = 384
```

### **Graph Limits** (`app/agents/state.py`)
```python
MAX_ITERATIONS = 5  # Previene bucles infinitos
```

---

## 📊 Traza ReAct

Cada ejecución genera una traza completa:
```json
{
  "query": "¿Qué es machine learning?",
  "timestamp": "2026-02-09T18:30:00",
  "iterations": 3,
  "final_answer": "Según el documento...",
  "trace": [
    {
      "step": 0,
      "agent": "coordinator",
      "timestamp": "2026-02-09T18:30:01",
      "thought": "Pregunta técnica...",
      "action": "search",
      "observation": "Decidido: search"
    },
    {
      "step": 1,
      "agent": "search",
      "timestamp": "2026-02-09T18:30:02",
      "thought": "Búsqueda en ChromaDB",
      "action": "retrieve",
      "observation": "5 documentos recuperados"
    },
    {
      "step": 2,
      "agent": "grader",
      "timestamp": "2026-02-09T18:30:03",
      "thought": "Evaluando relevancia",
      "action": "grade",
      "observation": "4 relevantes, 1 irrelevante"
    }
  ]
}
```

---

## 🧪 Testing

Ver `notebooks/04_langgraph_agents.ipynb`:

**Tests cubiertos:**
1. ✅ Coordinator - Análisis de queries
2. ✅ Múltiples tipos de consultas (técnicas, saludos, vagas)
3. ✅ Search Node - Recuperación de documentos
4. ✅ Grader - Evaluación de relevancia
5. ✅ Rewriter - Optimización de queries
6. ✅ Answer - Generación de respuestas
7. ✅ Grafo completo end-to-end
8. ✅ Conversaciones multi-turn
9. ✅ Límites de iteraciones
10. ✅ Exportación de trazas JSON

---

## 📦 Dependencias
```txt
langgraph==0.2.62
langchain==0.3.17
langchain-ollama==0.2.2
chromadb==0.5.23
sentence-transformers==3.4.0
```

---

## 🚀 Uso

### **Iniciar Ollama**
```bash
ollama serve
ollama pull llama3.2
```

### **Ejecutar Grafo**
```python
from app.agents.graph import run_graph

# Query simple
final_state = run_graph("¿Qué es inteligencia artificial?")
print(final_state['final_answer'])

# Acceder a traza
for step in final_state['trace']:
    print(f"{step['agent']}: {step['action']}")
```

### **Testing**
```bash
jupyter notebook notebooks/04_langgraph_agents.ipynb
```

---

## 📈 Métricas y Observabilidad

**Logging automático:**
- Cada nodo registra su ejecución
- Similarity scores de búsqueda
- Decisiones del Grader
- Reescrituras del Rewriter
- Respuestas generadas

**Traza exportable:**
- JSON completo de cada ejecución
- Guardado en `data/traces/`
- Útil para debugging y análisis

---

## 🔜 Próximas Mejoras

1. **API REST** (FastAPI) - Día 6-7
2. **Frontend React** - Día 8-10
3. **Streaming de respuestas** - Día 11
4. **Cache de resultados**
5. **Métricas de performance**
6. **A/B testing de prompts**

---

**Última actualización**: Febrero 9, 2026  
**Autor**: Darwin Arroyo - UCLA 
**Autor**: Julio Matheus - UCLA 