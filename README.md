# 🤖 Sistema RAG Multiagente con React + LangGraph

> Proyecto de Tesis - Universidad Centroccidental Lisandro Alvarado (UCLA)

Sistema de Recuperación Aumentada por Generación (RAG) con arquitectura multiagente para análisis de documentos académicos mediante agentes especializados que colaboran usando el patrón ReAct.

## 👥 Autores

- **Darwin Joel Arroyo Perez** - [@darwinjoelap](https://github.com/darwinjoelap) - Backend & Agentes
- **Julio Cesar Matheus** - [@juliomatheus](https://github.com/juliomatheus) - API & Frontend

**Tutor:** Dra. Maria Auxiliadora Perez  
**Universidad:** Universidad Centroccidental Lisandro Alvarado (UCLA)  
**Año:** 2026

---

## 📋 Descripción del Proyecto

Sistema inteligente que combina técnicas de RAG (Retrieval Augmented Generation) con una arquitectura multiagente desarrollada con LangGraph. Permite analizar documentos académicos mediante agentes especializados que colaboran usando el patrón ReAct para proporcionar respuestas contextuales y precisas.

### 🎯 Objetivos

- ✅ Implementar un sistema RAG multiagente utilizando tecnologías open-source
- ✅ Desarrollar agentes especializados con patrón ReAct (Reasoning + Acting)
- ✅ Crear sistema de recuperación con ChromaDB y embeddings semánticos
- 🚧 Desarrollar API REST con FastAPI (Día 6-7)
- 🚧 Crear interfaz web interactiva con React (Día 8-10)
- 🚧 Demostrar efectividad del sistema en análisis académico

---

## 🏗️ Arquitectura del Sistema

### Diagrama General
```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   React (WIP)    │◄────►│  FastAPI (WIP)   │◄────►│   LangGraph      │
│    Frontend      │      │      API         │      │  Multiagentes    │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                   │                          │
                                   ▼                          ▼
                          ┌─────────────────┐      ┌──────────────────┐
                          │    ChromaDB     │      │  Ollama Server   │
                          │  Vector Store   │      │   Llama 3.2      │
                          │  392 documentos │      │  Local LLM       │
                          └─────────────────┘      └──────────────────┘
```

### Flujo de Agentes (ReAct Pattern)
```
Usuario → Coordinator Agent
              ↓
        [Análisis ReAct]
         Thought: "Pregunta técnica..."
         Action: search
         Action Input: "query optimizada"
              ↓
         Search Node → ChromaDB
              ↓
        [5 documentos recuperados]
              ↓
         Grader Agent
              ↓
    ¿Documentos relevantes?
         /           \
       Sí            No
        ↓             ↓
   Answer Node   Rewriter Agent
        ↓             ↓
   [Respuesta]   [Nueva query]
                      ↓
                 Search Node (retry)
```

---

## 🧩 Componentes Implementados

### ✅ Backend RAG Multiagente (Días 1-5)

#### 1. **Base de Conocimiento** (Día 1-2)
- ✅ **Document Loader**: Carga PDFs desde `data/raw/`
- ✅ **Text Splitting**: RecursiveCharacterTextSplitter (500 chars, overlap 50)
- ✅ **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensiones)
- ✅ **Vector Store**: ChromaDB con persistencia local
- ✅ **Retriever**: Top-k=5, similarity_threshold=0.2
- ✅ **Estado**: 392 documentos indexados (2 libros de IA)

#### 2. **Sistema Multiagente LangGraph** (Día 3-5)

##### **Coordinator Agent** (`coordinator.py`)
- Cerebro del sistema
- Analiza queries usando patrón ReAct
- Decide acción: `search`, `answer`, o `clarify`
- Genera razonamiento explícito (Thought → Action → Observation)

##### **Search Node** (`search_node.py`)
- Recupera documentos relevantes de ChromaDB
- Integrado con `RetrieverService`
- Retorna top-5 documentos con similarity scores

##### **Grader Agent** (`grader.py`)
- Evalúa relevancia de cada documento recuperado
- Clasifica: `relevant` / `irrelevant`
- Decide next step: `answer` o `rewrite`

##### **Rewriter Agent** (`rewriter.py`)
- Optimiza queries que no dieron resultados
- Genera versión mejorada y más específica
- Reinicia búsqueda con nueva query

##### **Answer Node** (`answer_node.py`)
- Genera respuesta final al usuario
- Sintetiza información de documentos
- Cita fuentes originales
- Modos: contexto, directa, clarificación

##### **Graph Orchestration** (`graph.py`)
- Orquesta flujo completo de agentes
- Gestión de estado con `GraphState`
- Control de iteraciones (máx 5)
- Traza ReAct completa de ejecución

#### 3. **Servicios Core**
- ✅ `embeddings.py`: Servicio de embeddings
- ✅ `vector_store.py`: Gestión de ChromaDB
- ✅ `retriever.py`: Servicio de recuperación
- ✅ `llm.py`: Integración con Ollama
- ✅ `document_loader.py`: Carga de PDFs

### 🚧 API FastAPI (Día 6-7) - PENDIENTE

Endpoints planificados:
```
POST   /api/chat/              # Chat con el sistema
GET    /api/chat/history/{id}  # Historial de conversación
GET    /api/documents/stats    # Estadísticas del vector store
POST   /api/documents/upload   # Subir nuevos documentos
GET    /health                 # Health check
```

### 🚧 Frontend React (Día 8-10) - PENDIENTE

Componentes planificados:
- `ChatInterface`: Container principal
- `MessageList`: Lista de mensajes
- `InputBox`: Input del usuario
- `SourcesList`: Fuentes citadas
- `LoadingIndicator`: Estado de carga

---

## 🚀 Instalación

### Prerrequisitos

- **Python 3.11+**
- **Node.js 18+** (para frontend)
- **Ollama** instalado y corriendo
- **Git**

### 1. Clonar el Repositorio
```bash
git clone https://github.com/darwinjoelap/react-rag-multiagent-ucla.git
cd react-rag-multiagent-ucla
```

### 2. Configurar Backend
```bash
cd backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Configurar Ollama
```bash
# Iniciar servidor Ollama
ollama serve

# En otra terminal, descargar modelo
ollama pull llama3.2

# Verificar instalación
ollama list
```

### 4. Preparar Datos (Opcional - ya incluidos)
```bash
# Los documentos ya están indexados en data/vectorstore/
# Si quieres re-indexar:
cd backend
jupyter notebook

# Ejecutar notebooks en orden:
# 1. notebooks/01_document_loader.ipynb
# 2. notebooks/02_embeddings.ipynb
# 3. notebooks/03_retriever.ipynb
```

---

## 🎮 Uso Actual

### Probar Sistema Multiagente (Notebooks)
```bash
cd backend
jupyter notebook

# Abrir y ejecutar:
notebooks/04_langgraph_agents.ipynb
```

Este notebook contiene 15 tests completos:
1. ✅ Coordinator Agent individual
2. ✅ Múltiples tipos de queries
3. ✅ Search Node con ChromaDB
4. ✅ Grader Agent evaluation
5. ✅ Rewriter Agent optimization
6. ✅ Answer Node generation
7. ✅ Grafo completo (query simple)
8. ✅ Grafo completo (query técnica)
9. ✅ Conversación multi-turn
10. ✅ Límite de iteraciones
11. ✅ Exportación de trazas JSON

### Usar Programáticamente
```python
from app.agents.graph import run_graph

# Ejecutar query
final_state = run_graph("¿Qué es inteligencia artificial?")

# Ver respuesta
print(final_state['final_answer'])

# Ver documentos recuperados
for doc in final_state['retrieved_documents']:
    print(f"- {doc['metadata']['source']}: {doc['similarity']:.2f}")

# Ver traza completa
for step in final_state['trace']:
    print(f"{step['agent']}: {step['action']}")
```

### Ejemplos de Queries
```python
# Query técnica
run_graph("¿Qué es machine learning?")
# → Busca en docs → Grader → Respuesta con fuentes

# Saludo
run_graph("Hola, ¿cómo estás?")
# → Respuesta directa sin búsqueda

# Query ambigua
run_graph("Explícame eso")
# → Solicita clarificación

# Query de seguimiento
run_graph("¿Y cómo se relaciona con deep learning?")
# → Usa contexto previo → Búsqueda → Respuesta
```

---

## 📁 Estructura del Proyecto
```
react-rag-multiagent-ucla/
│
├── backend/
│   ├── app/
│   │   ├── agents/              # ⭐ Sistema multiagente
│   │   │   ├── state.py         # GraphState management
│   │   │   ├── coordinator.py   # CoordinatorAgent (ReAct)
│   │   │   ├── search_node.py   # SearchNode
│   │   │   ├── grader.py        # GraderAgent
│   │   │   ├── rewriter.py      # RewriterAgent
│   │   │   ├── answer_node.py   # AnswerNode
│   │   │   ├── graph.py         # LangGraph orchestration
│   │   │   └── prompts.py       # System prompts
│   │   │
│   │   ├── services/            # Servicios core
│   │   │   ├── embeddings.py
│   │   │   ├── vector_store.py
│   │   │   ├── retriever.py
│   │   │   ├── document_loader.py
│   │   │   └── llm.py
│   │   │
│   │   ├── core/
│   │   │   └── config.py
│   │   │
│   │   └── main.py              # FastAPI (WIP)
│   │
│   ├── data/
│   │   ├── raw/                 # PDFs originales
│   │   ├── vectorstore/         # ChromaDB (392 docs)
│   │   └── traces/              # Trazas de ejecución
│   │
│   ├── notebooks/               # ⭐ Testing & desarrollo
│   │   ├── 01_document_loader.ipynb
│   │   ├── 02_embeddings.ipynb
│   │   ├── 03_retriever.ipynb
│   │   └── 04_langgraph_agents.ipynb  # Tests completos
│   │
│   ├── requirements.txt
│   └── venv/
│
├── frontend/                    # React (WIP)
│
├── docs/
│   ├── BACKEND_ARCHITECTURE.md  # ⭐ Documentación técnica
│   └── HANDOFF_DAY_6.md         # ⭐ Handoff para Julio
│
└── README.md                    # Este archivo
```

---

## 🧪 Testing

### Tests Implementados (Notebooks)
```bash
cd backend
jupyter notebook notebooks/04_langgraph_agents.ipynb
```

**Cobertura:**
- ✅ Tests unitarios de cada agente
- ✅ Tests de integración del grafo completo
- ✅ Casos edge: queries vagas, límites, multi-turn
- ✅ Exportación de trazas para análisis

### Tests Futuros (Día 13-14)
```bash
# Backend (pytest)
cd backend
pytest tests/ -v

# Frontend (jest)
cd frontend
npm test
```

---

## 🛠️ Tecnologías Utilizadas

### Backend
- **LangGraph 0.2.62**: Orquestación de agentes
- **LangChain 0.3.17**: Framework de LLM
- **Ollama**: Servidor LLM local
- **Llama 3.2**: Modelo de lenguaje (3B parámetros)
- **ChromaDB 0.5.23**: Vector database
- **Sentence Transformers 3.4.0**: Embeddings (all-MiniLM-L6-v2)
- **FastAPI**: API REST (WIP)

### Frontend (Planificado)
- **React 18**: UI framework
- **Vite**: Build tool
- **TailwindCSS**: Styling
- **Axios**: HTTP client

### Infraestructura
- **Python 3.11**
- **Node.js 18+**
- **Jupyter**: Notebooks de desarrollo

---

## 📊 Estado del Proyecto

### ✅ Completado (Días 1-5)

| Componente | Estado | Descripción |
|------------|--------|-------------|
| Document Loader | ✅ | Carga y procesamiento de PDFs |
| Embeddings | ✅ | Generación de embeddings semánticos |
| Vector Store | ✅ | ChromaDB con 392 documentos |
| Retriever | ✅ | Búsqueda semántica (top-k=5) |
| Coordinator Agent | ✅ | Análisis ReAct de queries |
| Search Node | ✅ | Recuperación de documentos |
| Grader Agent | ✅ | Evaluación de relevancia |
| Rewriter Agent | ✅ | Optimización de queries |
| Answer Node | ✅ | Generación de respuestas |
| LangGraph | ✅ | Orquestación completa |
| Testing | ✅ | 15 notebooks de prueba |
| Documentación | ✅ | Backend completo documentado |

### 🚧 En Progreso (Días 6-15)

| Componente | Días | Responsable | Estado |
|------------|------|-------------|--------|
| API FastAPI | 6-7 | Julio | 🚧 Planificado |
| Frontend React | 8-10 | Julio | 🚧 Planificado |
| Features Avanzadas | 11-12 | Julio | 🚧 Planificado |
| Testing & Polish | 13-14 | Julio | 🚧 Planificado |
| Deploy | 15 | Ambos | 🚧 Planificado |

---

## 📚 Documentación Adicional

- **[Arquitectura Backend](docs/BACKEND_ARCHITECTURE.md)** - Documentación técnica completa del sistema multiagente
- **[Handoff Día 6](docs/HANDOFF_DAY_6.md)** - Guía detallada para continuar con API y Frontend
- **[Notebooks](backend/notebooks/)** - Jupyter notebooks con ejemplos y tests

---

## 🎓 Uso Académico

Este proyecto forma parte de una tesis de grado en Ingeniería en Informática en la UCLA. El objetivo es demostrar la viabilidad de sistemas RAG multiagente usando tecnologías open-source para análisis de documentos académicos.

### Contribuciones Principales

1. **Implementación de patrón ReAct** en sistema RAG
2. **Arquitectura multiagente** con LangGraph
3. **Integración de LLM local** (Ollama) sin dependencias de APIs comerciales
4. **Sistema completo end-to-end** desde indexación hasta respuesta

---

## 🚀 Roadmap

### Fase 1: Backend ✅ (Completado)
- [x] Sistema de carga de documentos
- [x] Generación de embeddings
- [x] Vector store con ChromaDB
- [x] Sistema multiagente con LangGraph
- [x] Patrón ReAct implementado
- [x] Testing completo en notebooks

### Fase 2: API 🚧 (Días 6-7)
- [ ] FastAPI endpoints
- [ ] CORS configuration
- [ ] Request/Response models
- [ ] Error handling
- [ ] API documentation (Swagger)

### Fase 3: Frontend 🚧 (Días 8-10)
- [ ] React setup
- [ ] Chat interface
- [ ] API integration
- [ ] Markdown rendering
- [ ] Source citations

### Fase 4: Features 🚧 (Días 11-12)
- [ ] Streaming responses
- [ ] Typing indicators
- [ ] Syntax highlighting
- [ ] Conversation export

### Fase 5: Deploy 🚧 (Días 13-15)
- [ ] Docker containerization
- [ ] Testing completo
- [ ] Documentation final
- [ ] Demo video

---

## 🤝 Contribución

Este es un proyecto académico. Si deseas contribuir:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- **Universidad Centroccidental Lisandro Alvarado (UCLA)**
- **Dra. Maria Auxiliadora Perez** - Tutora del proyecto
- **Comunidad LangChain/LangGraph** - Framework y documentación
- **Ollama Team** - LLM local open-source

---

## 📞 Contacto

### Darwin Joel Arroyo Perez
- Email: darwin@ucla.edu.ve
- GitHub: [@darwinjoelap](https://github.com/darwinjoelap)

### Julio Cesar Matheus
- Email: julio@ucla.edu.ve
- GitHub: [@juliomatheus](https://github.com/juliomatheus)

---

## 🔗 Enlaces Útiles

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Ollama Documentation](https://ollama.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

**Última actualización:** Febrero 9, 2026  
**Versión:** 1.0.0 (Backend completo)  
**Estado:** En desarrollo activo 🚀