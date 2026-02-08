# 🤖 Sistema RAG Multiagente con React + LangGraph

> Proyecto de Tesis - Universidad Lisandro Alvarado (UCLA)

Sistema de Recuperación Aumentada por Generación (RAG) con arquitectura multiagente para análisis académico inteligente.

## 👥 Autores

- **Darwin Joel Arroyo Perez** - [@darwinjoelap](https://github.com/darwinjoelap)
- **Julio Cesar Matheus** - [@juliomatheus](https://github.com/juliomatheus)

**Tutor:** Dra. Maria Auxiliadora Perez
**Universidad:** Universidad Centroccidental Lisandro Alvarado (UCLA)  
**Año:** 2025

---

## 📋 Descripción del Proyecto

Sistema inteligente que combina técnicas de RAG (Retrieval Augmented Generation) con una arquitectura multiagente desarrollada con LangGraph. Permite el análisis de documentos académicos mediante agentes especializados que colaboran para proporcionar respuestas contextuales y precisas.

### 🎯 Objetivos

- Implementar un sistema RAG multiagente utilizando tecnologías open-source
- Desarrollar agentes especializados para investigación, análisis y síntesis
- Crear una interfaz web intuitiva con React
- Demostrar la efectividad del sistema en análisis académico

---

## 🏗️ Arquitectura
```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   React     │◄────►│   FastAPI    │◄────►│  LangGraph  │
│  Frontend   │      │   Backend    │      │   Agents    │
└─────────────┘      └──────────────┘      └─────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │   ChromaDB   │
                     │ Vector Store │
                     └──────────────┘
                             │
                             ▼
                     ┌──────────────┐
                     │    Ollama    │
                     │  Llama 3.2   │
                     └──────────────┘
```

### Componentes Principales

1. **Frontend (React + Vite)**
   - Interfaz de usuario moderna y responsiva
   - Chat interactivo con el sistema
   - Visualización de documentos y resultados

2. **Backend (FastAPI)**
   - API RESTful para comunicación con el frontend
   - Gestión de documentos y vectorización
   - Orquestación de agentes

3. **Agentes LangGraph**
   - **Researcher Agent:** Búsqueda y recuperación de información
   - **Analyzer Agent:** Análisis profundo de contenido
   - **Synthesizer Agent:** Generación de respuestas coherentes

4. **Vector Store (ChromaDB)**
   - Almacenamiento de embeddings de documentos
   - Búsqueda por similitud semántica

5. **LLM (Ollama + Llama 3.2)**
   - Modelo de lenguaje local y gratuito
   - Procesamiento de lenguaje natural

---

## 🚀 Instalación

### Prerrequisitos

- Python 3.10+
- Node.js 18+
- Ollama instalado y corriendo
- Git

### 1. Clonar el repositorio
```bash
git clone https://github.com/TU_USUARIO/react-rag-multiagent-ucla.git
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

# Configurar variables de entorno
copy .env.example .env
# Editar .env con tus configuraciones
```

### 3. Configurar Ollama
```bash
# Descargar modelo Llama 3.2
ollama pull llama3.2

# Verificar que esté corriendo
ollama list
```

### 4. Configurar Frontend
```bash
cd ../frontend

# Instalar dependencias
npm install

# Copiar variables de entorno
copy .env.example .env.local
```

---

## 🎮 Uso

### Iniciar el Backend
```bash
cd backend
venv\Scripts\activate  # o source venv/bin/activate en Linux/Mac
uvicorn app.main:app --reload --port 8000
```

API disponible en: http://localhost:8000  
Documentación: http://localhost:8000/docs

### Iniciar el Frontend
```bash
cd frontend
npm run dev
```

Aplicación disponible en: http://localhost:5173

---

## 📁 Estructura del Proyecto
```
react-rag-multiagent-ucla/
├── frontend/           # Aplicación React
├── backend/           # API FastAPI + LangGraph
├── data/              # Documentos y vectorstore
├── notebooks/         # Jupyter notebooks
├── docs/              # Documentación
└── scripts/           # Scripts útiles
```

---

## 🧪 Testing
```bash
# Backend tests
cd backend
pytest

# Frontend tests
cd frontend
npm test
```

---

## 📚 Documentación Adicional

- [Arquitectura del Sistema](docs/architecture.md)
- [Referencia API](docs/api_reference.md)
- [Guía de Deployment](docs/deployment.md)

---

## 🛠️ Tecnologías Utilizadas

- **Frontend:** React, Vite, TailwindCSS
- **Backend:** FastAPI, Python
- **Agents:** LangGraph, LangChain
- **LLM:** Ollama (Llama 3.2)
- **Vector DB:** ChromaDB
- **Embeddings:** Sentence Transformers

---

## 📄 Licencia

MIT License - ver [LICENSE](LICENSE) para más detalles.

---

## 🙏 Agradecimientos

- Universidad Centroccidental Lisandro Alvarado (UCLA)
- Dra. Maria Auxiliadora Perez


---

## 📞 Contacto

¿Preguntas o sugerencias? Abre un [issue](https://github.com/darwinjoelap/react-rag-multiagent-ucla/issues) o contáctanos directamente.