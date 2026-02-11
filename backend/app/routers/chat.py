"""
Router para endpoints de chat
"""
from fastapi import APIRouter, HTTPException, status
from typing import Dict
import uuid
import logging
from datetime import datetime

from app.models.schemas import (
    ChatRequest, 
    ChatResponse, 
    Source,
    ConversationHistory,
    ConversationMessage,
    ErrorResponse
)
from app.agents.graph import run_graph

logger = logging.getLogger(__name__)

router = APIRouter()

# Almacenamiento en memoria de conversaciones
# TODO: Migrar a Redis o DB en producción
conversations: Dict[str, Dict] = {}


@router.post("/", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat(request: ChatRequest):
    """
    Endpoint principal de chat
    
    Procesa mensaje del usuario y retorna respuesta del sistema RAG multiagente.
    
    - **message**: Pregunta o mensaje del usuario (requerido)
    - **conversation_id**: ID de conversación existente (opcional)
    
    Returns:
    - **answer**: Respuesta generada por el sistema
    - **sources**: Documentos citados con similarity scores
    - **conversation_id**: ID de la conversación (nuevo o existente)
    - **timestamp**: Momento de la respuesta
    - **trace**: Traza ReAct completa (opcional, para debugging)
    """
    try:
        logger.info(f"📨 Nueva solicitud de chat: {request.message[:50]}...")
        
        # Generar o usar conversation_id
        conv_id = request.conversation_id or str(uuid.uuid4())
        
        # Ejecutar grafo multiagente
        logger.info(f"🤖 Ejecutando grafo para conversation_id: {conv_id}")
        final_state = run_graph(request.message)
        
        logger.info(f"✅ Grafo ejecutado. Iteraciones: {final_state.get('iteration', 0)}")
        
        # Formatear fuentes
        sources = []
        for doc in final_state.get("retrieved_documents", []):
            sources.append(Source(
                document=doc.get("document", "")[:300] + "...",  # Limitar longitud
                source=doc.get("metadata", {}).get("source", "unknown").split("\\")[-1],  # Solo filename
                similarity=round(doc.get("similarity", 0.0), 4)
            ))
        
        # Guardar en historial
        if conv_id not in conversations:
            conversations[conv_id] = {
                "messages": [],
                "created_at": datetime.now().isoformat(),
            }
        
        conversations[conv_id]["messages"].extend([
            {
                "role": "user",
                "content": request.message,
                "timestamp": datetime.now().isoformat()
            },
            {
                "role": "assistant",
                "content": final_state.get("final_answer", ""),
                "timestamp": datetime.now().isoformat()
            }
        ])
        conversations[conv_id]["updated_at"] = datetime.now().isoformat()
        
        # Preparar respuesta
        response = ChatResponse(
            answer=final_state.get("final_answer", ""),
            sources=sources,
            conversation_id=conv_id,
            timestamp=datetime.now().isoformat(),
            trace=None  # Incluir solo si se pide explícitamente
        )
        
        logger.info(f"✅ Respuesta enviada. Fuentes: {len(sources)}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error en chat: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error procesando mensaje: {str(e)}"
        )


@router.get("/history/{conversation_id}", response_model=ConversationHistory)
async def get_history(conversation_id: str):
    """
    Obtener historial de una conversación
    
    - **conversation_id**: ID de la conversación
    
    Returns:
    - Historial completo de mensajes con timestamps
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversación {conversation_id} no encontrada"
        )
    
    conv = conversations[conversation_id]
    
    return ConversationHistory(
        conversation_id=conversation_id,
        messages=[ConversationMessage(**msg) for msg in conv["messages"]],
        created_at=conv["created_at"],
        updated_at=conv.get("updated_at", conv["created_at"])
    )


@router.delete("/history/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_history(conversation_id: str):
    """
    Eliminar historial de una conversación
    
    - **conversation_id**: ID de la conversación a eliminar
    """
    if conversation_id not in conversations:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Conversación {conversation_id} no encontrada"
        )
    
    del conversations[conversation_id]
    logger.info(f"🗑️ Conversación {conversation_id} eliminada")
    return None


@router.get("/conversations", response_model=list)
async def list_conversations():
    """
    Listar todas las conversaciones activas
    
    Returns:
    - Lista de IDs de conversación con metadata básica
    """
    result = []
    for conv_id, conv_data in conversations.items():
        result.append({
            "conversation_id": conv_id,
            "message_count": len(conv_data["messages"]),
            "created_at": conv_data["created_at"],
            "updated_at": conv_data.get("updated_at", conv_data["created_at"])
        })
    
    return result