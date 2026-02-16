# 🤖 Sistema RAG Multi-Agente con Patrón ReAct + LangGraph

> **Proyecto de Tesis** - Universidad Centroccidental Lisandro Alvarado (UCLA)  
> **Sistema de Recuperación Aumentada por Generación (RAG)** con arquitectura multi-agente para análisis de documentos académicos

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2.62-green.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Latest-009688.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18-61DAFB.svg)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 👥 Autores

**Darwin Joel Arroyo Perez**   
**Julio Cesar Matheus Arroyo**
**Tutor:** Dra. Maria Auxiliadora Perez  
**Universidad:** Universidad Centroccidental Lisandro Alvarado (UCLA)  
**Año:** 2026

---

## 📋 Descripción del Proyecto

Sistema inteligente que combina **RAG (Retrieval-Augmented Generation)** con una arquitectura **multi-agente** desarrollada con **LangGraph**. Implementa el **patrón ReAct** (Reasoning and Acting) para responder preguntas complejas sobre documentos académicos de Inteligencia Artificial y Machine Learning.

### 🎯 Características Principales

✅ **5 Agentes Especializados** colaborando en flujo orquestado  
✅ **Patrón ReAct** (Thought → Action → Observation → Decision)  
✅ **Auto-corrección** mediante reformulación de queries (hasta 2 reintentos)  
✅ **466 documentos** académicos indexados semánticamente  
✅ **LLM Local** (Llama 3.2 vía Ollama) - sin dependencias de APIs comerciales  
✅ **Visualización en tiempo real** del flujo con grafos Mermaid  
✅ **Streaming SSE** para respuestas progresivas  
✅ **Trazas ReAct auditables** completas  

### 🏆 Resultados

- **Precisión:** 85% (vs 60% baseline monolítico)
- **Manejo fuera de dominio:** 100% (admite limitaciones sin alucinar)
- **Tiempo de respuesta:** ~30-50 segundos (optimizado desde 4.6 minutos)
- **Precisión en queries ambiguas:** 80% (vs 45% baseline)

---

## 🏗️ Arquitectura del Sistema

### Diagrama General

```
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│   React + TS     │◄────►│     FastAPI      │◄────►│   LangGraph      │
│   Frontend       │      │   Streaming SSE  │      │  Multi-Agentes   │
│   + Mermaid.js   │      │   CORS Enabled   │      │  (5 Agentes)     │
└──────────────────┘      └──────────────────┘      └──────────────────┘
                                   │                          │
                                   ▼                          ▼
                          ┌─────────────────┐      ┌──────────────────┐
                          │    ChromaDB     │      │  Ollama Server   │
                          │  Vector Store   │      │   Llama 3.2      │
                          │  466 documentos │      │  Local LLM       │
                          └─────────────────┘      └──────────────────┘
```

### Flujo de Agentes (Patrón ReAct)

```
Usuario → 📊 Coordinator Agent (ReAct)
              │
              ├─ Thought: "Necesito buscar información..."
              ├─ Action: search / answer / rewrite
              ├─ Observation: "5 docs recuperados, sim=0.35"
              └─ Decision: → siguiente nodo
                    │
            ┌───────┴────────┐
            ▼                ▼
      🔍 Search Agent    💬 Answer Agent
            │                    ↑
            ▼                    │
      ✅ Grader Agent            │
            │                    │
    ┌───────┴────────┐           │
    ▼                ▼           │
Relevante      Irrelevante       │
    │                │           │
    └────────────────┤           │
                     ▼           │
              🔄 Rewriter Agent  │
                     │           │
              (retry < 2) ───────┘
                     │
              (retry >= 2) → Answer forzado
```

---

## 🧩 Los 5 Agentes del Sistema

### 1️⃣ **Coordinator Agent** (ReAct Pattern)
- **Responsabilidad:** Analiza queries y decide estrategia
- **Decisiones:** `search`, `answer`, `clarify`
- **Características:** Expansión automática de queries ambiguas
- **Prompt:** CoT (Chain-of-Thought) explícito

### 2️⃣ **Search Agent** (Tool Use + RAG)
- **Responsabilidad:** Recuperación semántica de documentos
- **Tecnología:** ChromaDB + sentence-transformers (384D)
- **Parámetros:** Top-K=5, Threshold=0.2
- **Output:** Documentos con scores de similitud

### 3️⃣ **Grader Agent** (Multi-Agent)
- **Responsabilidad:** Evaluación de relevancia de documentos
- **Método:** Similarity score (threshold 0.25)
- **Optimización:** LLM-free (< 1ms vs 38s con LLM)
- **Decisión:** `answer` (si relevantes) o `rewrite` (si irrelevantes)

### 4️⃣ **Rewriter Agent** (Multi-Agent + ReAct Loop)
- **Responsabilidad:** Reformulación de queries pobres
- **Estrategias:** Expansión, generalización, sinónimos
- **Límite:** Máximo 2 reintentos (previene loops infinitos)
- **Output:** Query optimizada para nueva búsqueda

### 5️⃣ **Answer Agent** (RAG Generation)
- **Responsabilidad:** Síntesis de respuesta final
- **LLM:** Llama 3.2 (Ollama local)
- **Parámetros:** Temperatura 0.3, Max tokens 1024
- **Modo:** RAG puro (solo información del contexto)

---

## 🚀 Inicio Rápido

### Prerrequisitos

Asegúrate de tener instalado:

- **Python 3.11+** ([Descargar](https://www.python.org/downloads/))
- **Node.js 18+** ([Descargar](https://nodejs.org/))
- **Ollama** ([Descargar](https://ollama.com/download))
- **Git**

### Instalación Paso a Paso

#### 1️⃣ **Clonar el Repositorio**

```bash
git clone https://github.com/darwinjoelap/react-rag-multiagent-ucla.git
cd react-rag-multiagent-ucla
```

#### 2️⃣ **Configurar Backend**

```bash
# Navegar a backend
cd backend

# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt --break-system-packages  # Si es necesario

# Volver a la raíz
cd ..
```

#### 3️⃣ **Configurar Ollama**

```bash
# TERMINAL 1: Iniciar servidor Ollama (dejar corriendo)
ollama serve

# TERMINAL 2: Descargar modelo Llama 3.2
ollama pull llama3.2

# Verificar instalación
ollama list
# Deberías ver: llama3.2:latest
```

#### 4️⃣ **Verificar Base de Conocimiento**

```bash
# Los 466 documentos ya están indexados en data/vectorstore/
# Para verificar:
cd backend
python -c "from app.services.vector_store import VectorStoreService; vs = VectorStoreService(); print(f'Documentos indexados: {vs.collection.count()}')"

# Salida esperada: Documentos indexados: 466
```

#### 5️⃣ **Iniciar Backend (FastAPI)**

```bash
# TERMINAL 3: Desde la raíz del proyecto
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# ✅ Backend corriendo en: http://localhost:8000
# ✅ Docs interactivos: http://localhost:8000/docs
```

#### 6️⃣ **Iniciar Frontend (React)**

```bash
# TERMINAL 4: Desde la raíz del proyecto
cd frontend

# Instalar dependencias (solo primera vez)
npm install

# Iniciar servidor de desarrollo
npm run dev

# ✅ Frontend corriendo en: http://localhost:3000
```

---

## 🎮 Uso del Sistema

### Interfaz Web (Recomendado)

1. **Abrir navegador:** http://localhost:3000
2. **Escribir query:** Ejemplo: "¿Qué es un agente inteligente?"
3. **Observar:**
   - **Panel izquierdo:** Grafo del flujo en tiempo real
   - **Panel central:** Respuesta streaming
   - **Panel derecho:** Timeline ReAct con trazas
4. **Métricas:** Tiempo, documentos recuperados, iteraciones

### API REST (Programático)

#### **Endpoint Principal: Chat Streaming**

```bash
# cURL Example
curl -X POST "http://localhost:8000/api/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "¿Qué es machine learning?",
    "history": []
  }'
```

#### **Python SDK**

```python
import requests

url = "http://localhost:8000/api/chat/stream"
payload = {
    "query": "¿Qué es un agente inteligente?",
    "history": []
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
```

#### **Otros Endpoints**

```bash
# Health Check
GET http://localhost:8000/health

# Estadísticas de documentos
GET http://localhost:8000/api/documents/stats

# Historial de conversación
GET http://localhost:8000/api/chat/history/{conversation_id}
```

---

## 📊 Ejemplos de Queries

### ✅ **Caso 1: Query Exitosa Directa**

```json
{
  "query": "¿Qué es un agente inteligente y cuáles son sus componentes?",
  "history": []
}
```

**Resultado esperado:**
- ✅ 5 documentos de `Módulo2_Agentes.pdf`
- ✅ Similitud máxima: ~0.39
- ✅ Iteraciones: 1 (éxito directo)
- ⏱️ Tiempo: ~33 segundos

---

### 🔄 **Caso 2: Auto-corrección (Query Ambigua)**

```json
{
  "query": "CNN",
  "history": []
}
```

**Flujo esperado:**
1. Primera búsqueda → resultados pobres
2. Grader rechaza → `rewrite`
3. Rewriter: "CNN" → "Redes Neuronales Convolucionales"
4. Segunda búsqueda → éxito
5. Respuesta sobre redes convolucionales

---

### ⚠️ **Caso 3: Fuera de Dominio**

```json
{
  "query": "¿Cuál es el precio del Bitcoin?",
  "history": []
}
```

**Comportamiento esperado:**
- ❌ Similitud negativa (-0.17 → -0.09 → +0.09)
- 🔄 2 reintentos de reformulación
- ✅ Límite alcanzado → respuesta honesta
- 💬 "No tengo información sobre Bitcoin en mi base de conocimiento..."
- 🎯 **No alucina información**

---

### 🎓 **Caso 4: Query Técnica**

```json
{
  "query": "Explica cómo funciona el algoritmo de backpropagation",
  "history": []
}
```

**Resultado esperado:**
- Recupera de `Redes Neuronales Artificiales.pdf` y `Russell-Norvig.pdf`
- Posible reescritura si threshold estricto
- Respuesta técnica con fundamentos

---

## 📁 Estructura del Proyecto

```
react-rag-multiagent-ucla/
│
├── backend/                     # Backend Python
│   ├── app/
│   │   ├── agents/              # ⭐ Sistema Multi-Agente
│   │   │   ├── state.py         # GraphState (messages, documents, retry_count)
│   │   │   ├── coordinator.py   # Coordinator Agent (ReAct)
│   │   │   ├── search_node.py   # Search Agent (RAG)
│   │   │   ├── grader.py        # Grader Agent (Multi-Agent)
│   │   │   ├── rewriter.py      # Rewriter Agent (ReAct Loop)
│   │   │   ├── answer_node.py   # Answer Agent (Generation)
│   │   │   ├── graph.py         # LangGraph Orchestration
│   │   │   └── prompts.py       # System Prompts
│   │   │
│   │   ├── services/            # Servicios Core
│   │   │   ├── embeddings.py    # Embedding Service (MiniLM-L6-v2)
│   │   │   ├── vector_store.py  # ChromaDB Management
│   │   │   ├── retriever.py     # Retrieval Service
│   │   │   ├── document_loader.py  # PDF Loader
│   │   │   └── llm.py           # Ollama Integration
│   │   │
│   │   ├── routers/             # FastAPI Routers
│   │   │   └── chat.py          # Chat endpoints
│   │   │
│   │   ├── core/
│   │   │   └── config.py        # Configuration
│   │   │
│   │   └── main.py              # FastAPI App
│   │
│   ├── data/
│   │   ├── raw/                 # PDFs originales (10 archivos)
│   │   └── vectorstore/         # ChromaDB (466 documentos)
│   │
│   ├── requirements.txt         # Python dependencies
│   └── venv/                    # Virtual environment
│
├── frontend/                    # Frontend React + TypeScript
│   ├── src/
│   │   ├── components/          # React Components
│   │   ├── hooks/               # Custom Hooks
│   │   ├── services/            # API Services
│   │   └── App.tsx              # Main App
│   │
│   ├── package.json
│   └── node_modules/
│
├── docs/                        # Documentación
│   ├── Informe_RAG_Multi_Agente_UCLA.md
│   ├── Presentacion_RAG_Multi_Agente_20min.md
│   ├── Guion_Presentacion_20min.md
│   └── Trazas_amplias.png
│
├── README.md                    # Este archivo
└── .gitignore
```

---

## 🛠️ Stack Tecnológico

### **Backend**

| Componente | Tecnología | Versión | Justificación |
|------------|------------|---------|---------------|
| Orquestación | LangGraph | 0.2.62 | Flujos condicionales complejos, debugging granular |
| LLM | Llama 3.2 (Ollama) | Latest | Ejecución local, costo cero, reproducibilidad |
| Vector Store | ChromaDB | 0.5.23 | Persistencia local, fácil integración |
| Embeddings | sentence-transformers | 3.4.0 | MiniLM-L6-v2: 384D, multilingüe, rápido |
| Backend API | FastAPI | Latest | Async nativo, streaming SSE, docs automáticas |
| PDF Processing | pypdf | Latest | Extracción de texto de PDFs |

### **Frontend**

| Componente | Tecnología | Justificación |
|------------|------------|---------------|
| Framework | React 18 + TypeScript | Type safety, componentes reutilizables |
| Build Tool | Vite | Dev server rápido, HMR optimizado |
| Estilos | Tailwind CSS | Utility-first, responsive, personalización rápida |
| Visualización | Mermaid.js | Grafos de flujo dinámicos, exportación SVG |
| HTTP Client | Axios | Manejo de streaming, interceptors |

### **Base de Conocimiento**

- **Total:** 466 chunks indexados
- **Fuentes:** 10 documentos PDF académicos
- **Temas:** IA, Machine Learning, Redes Neuronales, Agentes, Búsqueda Heurística
- **Dimensión embeddings:** 384
- **Modelo:** sentence-transformers/all-MiniLM-L6-v2

---

## 🧪 Testing y Diagnóstico

### **Verificar Estado del Sistema**

```bash
# Desde backend/ con venv activado
python test_diagnostico_corregido.py
```

**Output esperado:**
```
📊 Total de chunks indexados: 466
📁 Total de fuentes únicas: 10
✅ Archivos nuevos encontrados: 9/9
```

### **Re-indexar Documentos (Si es necesario)**

```bash
# Desde la raíz del proyecto
python index_documents.py --reindex
```

---

## 🎯 Métricas y Rendimiento

### **Optimizaciones Implementadas**

| Optimización | Impacto | Antes | Después |
|--------------|---------|-------|---------|
| LLM-free Grader | 🟢 Crítico | 38s | <1s |
| Threshold 0.25 | 🟡 Alto | Precisión 60% | 85% |
| Max tokens 1024 | 🟡 Medio | Respuestas largas | Concisas |
| Chunking 1000 chars | 🟡 Medio | 800 chunks | 466 chunks |
| Max retries 2 | 🟢 Alto | Loops infinitos | Control estricto |

**Resultado:** De 4.6 minutos → 50 segundos (82% reducción)

### **Comparativa: Multi-Agente vs Baseline**

| Métrica | Baseline Monolítico | Multi-Agente ReAct |
|---------|--------------------|--------------------|
| Precisión (queries claras) | 75% | **85%** ⬆️ +10% |
| Precisión (queries ambiguas) | 45% | **80%** ⬆️ +35% |
| Manejo fuera de dominio | 0% (alucina) | **100%** (honesto) |
| Latencia promedio | 15s | 30s ⬇️ +15s |
| Transparencia | Nula (caja negra) | **Alta** (trazas completas) |
| Auto-corrección | No | **Sí** (2 reintentos) |

---

## 📚 Documentación Adicional

- **[Informe Completo](docs/Informe_RAG_Multi_Agente_UCLA.md)** - Documento académico de 4 páginas
- **[Trazas Visuales](docs/Trazas_amplias.png)** - Resumen de 4 casos de prueba

---

## ⚙️ Configuración Avanzada

### **Variables de Entorno (Opcional)**

Crear archivo `.env` en `backend/`:

```env
# LLM Configuration
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1024

# Vector Store
CHROMA_PERSIST_DIR=../data/vectorstore
COLLECTION_NAME=ucla_documents

# Retriever
TOP_K=5
SIMILARITY_THRESHOLD=0.2

# Grader
GRADER_THRESHOLD=0.25

# Rewriter
MAX_RETRIES=2

# API
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

### **Re-indexar con Parámetros Personalizados**

```python
# Editar index_documents.py

index_documents(
    data_dir="data/raw",
    chunk_size=1000,      # Tamaño de chunks
    overlap=200,          # Overlap entre chunks
    reindex=True          # Borrar colección existente
)
```

---

## 🐛 Solución de Problemas

### **Error: "Ollama server not running"**

```bash
# Verificar si Ollama está corriendo
curl http://localhost:11434/api/tags

# Si no responde, iniciar Ollama
ollama serve
```

### **Error: "Collection not found"**

```bash
# Re-indexar documentos
python index_documents.py --reindex
```

### **Error: "CORS policy blocking"**

```python
# En backend/app/main.py, verificar CORS:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### **Frontend no conecta con backend**

```bash
# Verificar que backend está en puerto 8000
curl http://localhost:8000/health

# Verificar VITE_API_URL en frontend/.env
VITE_API_URL=http://localhost:8000
```

---

## 🎓 Uso Académico

Este proyecto forma parte del primer laboratorio de la Maestría en ciencias de la computación (UCLA)

### **Contribuciones Principales**

1. ✅ Implementación práctica del **patrón ReAct** en sistema RAG
2. ✅ Arquitectura **multi-agente** con LangGraph (5 agentes especializados)
3. ✅ Integración de **LLM local** (Ollama) sin dependencias de APIs comerciales
4. ✅ Sistema completo **end-to-end** con visualización en tiempo real
5. ✅ **Optimización de rendimiento** (82% reducción en latencia)
6. ✅ **Manejo robusto** de queries fuera de dominio (no alucina)


---

## 🤝 Contribución

Este es un proyecto académico. Contribuciones son bienvenidas:

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/NuevaCaracteristica`
3. Commit: `git commit -m 'Add: Nueva característica'`
4. Push: `git push origin feature/NuevaCaracteristica`
5. Abre un Pull Request

---

## 📄 Licencia

Este proyecto está bajo la **Licencia MIT**. Ver archivo [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- **Universidad Centroccidental Lisandro Alvarado (UCLA)** - Apoyo institucional
- **Dra. Maria Auxiliadora Perez** - Tutoría del proyecto
- **LangChain/LangGraph Community** - Framework y documentación
- **Ollama Team** - LLM local open-source de calidad
- **ChromaDB Team** - Vector database eficiente

---

## 📞 Contacto

**Darwin Joel Arroyo Perez**  
📧 Email: darwinjoelap@gmail.com  
🐙 GitHub: [@darwinjoelap](https://github.com/darwinjoelap)  
🎓 Universidad: UCLA, Venezuela

**Julio Cesar Matheus Arroyo**  
📧 Email: juliomatheus@gmail.com   
🎓 Universidad: UCLA, Venezuela



---

## 🔗 Enlaces Útiles

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Documentation](https://python.langchain.com/)
- [Ollama Documentation](https://ollama.com/docs)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)
- [Mermaid.js Documentation](https://mermaid.js.org/)

---

## 📊 Estado del Proyecto

**Versión:** 2.0.0 (Sistema Completo)  
**Estado:** ✅ Completado y funcional  
**Última actualización:** Febrero 15, 2026  
**Líneas de código:** ~5,000 (Python) + ~3,000 (TypeScript)  
**Documentos indexados:** 466 chunks de 10 PDFs académicos  

---

<div align="center">
**Para la comunidad académica open-source** 🎓

</div>
