from typing import List
import logging
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Servicio para generar embeddings de texto"""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2"
        #model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ):
        """
        Inicializar servicio de embeddings
        
        Args:
            model_name: Nombre del modelo de embeddings.
                       Por defecto usa un modelo multilingüe que soporta español.
        """
        logger.info(f"Cargando modelo de embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Modelo cargado. Dimensión: {self.dimension}")
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generar embedding para un texto
        
        Args:
            text: Texto a embebir
            
        Returns:
            Lista de floats representando el embedding
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generando embedding: {str(e)}")
            raise
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generar embeddings para múltiples textos
        
        Args:
            texts: Lista de textos
            
        Returns:
            Lista de embeddings
        """
        try:
            logger.info(f"Generando embeddings para {len(texts)} textos")
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=True
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error generando embeddings: {str(e)}")
            raise
    
    def embed_query(self, query: str) -> List[float]:
        """
        Generar embedding para una consulta (query)
        Alias de embed_text para claridad semántica
        
        Args:
            query: Consulta del usuario
            
        Returns:
            Embedding de la consulta
        """
        return self.embed_text(query)
    
    def cosine_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """
        Calcular similitud coseno entre dos embeddings
        
        Args:
            embedding1: Primer embedding
            embedding2: Segundo embedding
            
        Returns:
            Similitud coseno (0-1)
        """
        e1 = np.array(embedding1)
        e2 = np.array(embedding2)
        
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
    
    def get_model_info(self) -> dict:
        """Obtener información del modelo"""
        return {
            "model_name": self.model._model_card_vars.get("model_name", "unknown"),
            "dimension": self.dimension,
            "max_seq_length": self.model.max_seq_length
        }

# Instancia global
embedding_service = EmbeddingService()