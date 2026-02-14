"""
Pydantic models para request/response de la API
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import datetime


class ChatRequest(BaseModel):
    """Request para endpoint de chat con soporte multi-turno"""
    message: str = Field(..., min_length=1, max_length=2000, description="Mensaje del usuario")
    conversation_id: Optional[str] = Field(None, description="ID de conversación existente")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None, 
        description="Historial de mensajes previos para contexto multi-turno"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "¿Y eso es lo mismo que deep learning?",
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "conversation_history": [
                    {"role": "user", "content": "¿Qué es machine learning?", "timestamp": "2026-02-13T22:00:00"},
                    {"role": "assistant", "content": "El machine learning es...", "timestamp": "2026-02-13T22:00:05"}
                ]
            }
        }


class Source(BaseModel):
    """Documento fuente citado en la respuesta"""
    document: str = Field(..., description="Fragmento del documento")
    source: str = Field(..., description="Nombre del archivo fuente")
    similarity: float = Field(..., ge=0.0, le=1.0, description="Score de similitud")
    
    class Config:
        json_schema_extra = {
            "example": {
                "document": "La inteligencia artificial es el campo de estudio...",
                "source": "Russell-Norvig.pdf",
                "similarity": 0.85
            }
        }


class TraceStep(BaseModel):
    """Paso individual en la traza ReAct"""
    step: int
    agent: str
    timestamp: str
    thought: str
    action: str
    observation: str


class ChatResponse(BaseModel):
    """Response del endpoint de chat"""
    answer: str = Field(..., description="Respuesta generada por el sistema")
    sources: List[Source] = Field(default_factory=list, description="Fuentes citadas")
    conversation_id: str = Field(..., description="ID de la conversación")
    timestamp: str = Field(..., description="Timestamp de la respuesta")
    trace: Optional[List[TraceStep]] = Field(None, description="Traza ReAct (opcional)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Según Russell-Norvig.pdf, la inteligencia artificial es...",
                "sources": [
                    {
                        "document": "La IA es el campo de estudio...",
                        "source": "Russell-Norvig.pdf",
                        "similarity": 0.85
                    }
                ],
                "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-02-10T10:30:00",
                "trace": None
            }
        }


class ConversationMessage(BaseModel):
    """Mensaje individual en historial"""
    role: str = Field(..., description="Rol: 'user' o 'assistant'")
    content: str
    timestamp: str


class ConversationHistory(BaseModel):
    """Historial completo de conversación"""
    conversation_id: str
    messages: List[ConversationMessage]
    created_at: str
    updated_at: str


class DocumentStats(BaseModel):
    """Estadísticas del vector store"""
    total_documents: int = Field(..., description="Total de documentos indexados")
    unique_sources: int = Field(..., description="Número de archivos únicos")
    file_types: Dict[str, int] = Field(..., description="Documentos por tipo de archivo")
    last_updated: Optional[str] = Field(None, description="Última actualización")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_documents": 392,
                "unique_sources": 2,
                "file_types": {"pdf": 392},
                "last_updated": "2026-02-09T18:00:00"
            }
        }


class HealthCheck(BaseModel):
    """Response del health check"""
    status: str
    ollama_status: str
    chromadb_status: str
    timestamp: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "ollama_status": "connected",
                "chromadb_status": "connected",
                "timestamp": "2026-02-10T10:30:00"
            }
        }


class ErrorResponse(BaseModel):
    """Response de error estándar"""
    detail: str
    error_code: Optional[str] = None
    timestamp: str

# ==================== STREAMING EVENTS ====================

class StreamEventBase(BaseModel):
    """Modelo base para eventos de streaming"""
    event_type: str = Field(..., description="Tipo de evento")
    timestamp: str = Field(..., description="Timestamp del evento")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "node_start",
                "timestamp": "2026-02-12T10:30:00"
            }
        }


class NodeStartEvent(StreamEventBase):
    """Evento: Inicio de nodo"""
    event_type: str = "node_start"
    node_name: str = Field(..., description="Nombre del nodo (coordinator, search, grader, etc)")
    iteration: int = Field(..., description="Número de iteración actual")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "node_start",
                "node_name": "coordinator",
                "iteration": 1,
                "timestamp": "2026-02-12T10:30:00"
            }
        }


class NodeEndEvent(StreamEventBase):
    """Evento: Fin de nodo"""
    event_type: str = "node_end"
    node_name: str = Field(..., description="Nombre del nodo")
    iteration: int = Field(..., description="Número de iteración actual")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "node_end",
                "node_name": "coordinator",
                "iteration": 1,
                "timestamp": "2026-02-12T10:30:01"
            }
        }


class ThoughtEvent(StreamEventBase):
    """Evento: Pensamiento del coordinador (ReAct)"""
    event_type: str = "thought"
    thought: str = Field(..., description="Pensamiento del agente")
    action: str = Field(..., description="Acción decidida (search/answer/clarify)")
    iteration: int = Field(..., description="Número de iteración")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "thought",
                "thought": "El usuario pregunta sobre IA, necesito buscar en la base de conocimiento",
                "action": "search",
                "iteration": 1,
                "timestamp": "2026-02-12T10:30:00"
            }
        }


class DocumentsRetrievedEvent(StreamEventBase):
    """Evento: Documentos recuperados del vector store"""
    event_type: str = "documents_retrieved"
    document_count: int = Field(..., description="Número de documentos recuperados")
    sources: List[str] = Field(..., description="Lista de fuentes")
    iteration: int = Field(..., description="Número de iteración")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "documents_retrieved",
                "document_count": 5,
                "sources": ["Russell-Norvig.pdf", "Machine Learning.pdf"],
                "iteration": 1,
                "timestamp": "2026-02-12T10:30:02"
            }
        }


class GradingResultEvent(StreamEventBase):
    """Evento: Resultado del grader"""
    event_type: str = "grading_result"
    relevant_count: int = Field(..., description="Documentos relevantes")
    total_count: int = Field(..., description="Total de documentos evaluados")
    decision: str = Field(..., description="Decisión: 'proceed' o 'rewrite'")
    iteration: int = Field(..., description="Número de iteración")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "grading_result",
                "relevant_count": 3,
                "total_count": 5,
                "decision": "proceed",
                "iteration": 1,
                "timestamp": "2026-02-12T10:30:03"
            }
        }


class RewriteEvent(StreamEventBase):
    """Evento: Query reescrita"""
    event_type: str = "rewrite"
    original_query: str = Field(..., description="Query original")
    rewritten_query: str = Field(..., description="Query reescrita")
    iteration: int = Field(..., description="Número de iteración")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "rewrite",
                "original_query": "¿Qué es IA?",
                "rewritten_query": "¿Cuál es la definición formal de inteligencia artificial según Russell y Norvig?",
                "iteration": 2,
                "timestamp": "2026-02-12T10:30:04"
            }
        }


class FinalAnswerEvent(StreamEventBase):
    """Evento: Respuesta final generada"""
    event_type: str = "final_answer"
    answer: str = Field(..., description="Respuesta final del sistema")
    sources: List[Source] = Field(..., description="Fuentes citadas")
    total_iterations: int = Field(..., description="Iteraciones totales")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "final_answer",
                "answer": "La inteligencia artificial es...",
                "sources": [],
                "total_iterations": 2,
                "timestamp": "2026-02-12T10:30:05"
            }
        }


class ErrorEvent(StreamEventBase):
    """Evento: Error durante el procesamiento"""
    event_type: str = "error"
    error_message: str = Field(..., description="Mensaje de error")
    node_name: Optional[str] = Field(None, description="Nodo donde ocurrió el error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "error",
                "error_message": "Error conectando con Ollama",
                "node_name": "coordinator",
                "timestamp": "2026-02-12T10:30:00"
            }
        }


class DoneEvent(StreamEventBase):
    """Evento: Procesamiento completado"""
    event_type: str = "done"
    success: bool = Field(..., description="Si el procesamiento fue exitoso")
    total_time_seconds: float = Field(..., description="Tiempo total de procesamiento")
    
    class Config:
        json_schema_extra = {
            "example": {
                "event_type": "done",
                "success": True,
                "total_time_seconds": 125.3,
                "timestamp": "2026-02-12T10:32:05"
            }
        }