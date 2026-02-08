from typing import List, Optional
from pathlib import Path
import logging
from pypdf import PdfReader
from docx import Document as DocxDocument
# Línea 6 - CAMBIAR
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Línea 7 - CAMBIAR  
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class DocumentLoader:
    """Carga y procesa documentos PDF, DOCX y TXT"""
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Cargar documento PDF"""
        try:
            logger.info(f"Cargando PDF: {file_path}")
            reader = PdfReader(file_path)
            
            text = ""
            metadata = {
                "source": file_path,
                "file_type": "pdf",
                "total_pages": len(reader.pages)
            }
            
            # Extraer texto de todas las páginas
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                text += f"\n\n--- Página {page_num} ---\n\n{page_text}"
            
            # Dividir en chunks
            documents = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            
            logger.info(f"PDF procesado: {len(documents)} chunks creados")
            return documents
            
        except Exception as e:
            logger.error(f"Error cargando PDF {file_path}: {str(e)}")
            raise
    
    def load_docx(self, file_path: str) -> List[Document]:
        """Cargar documento DOCX"""
        try:
            logger.info(f"Cargando DOCX: {file_path}")
            doc = DocxDocument(file_path)
            
            text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
            metadata = {
                "source": file_path,
                "file_type": "docx"
            }
            
            documents = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            
            logger.info(f"DOCX procesado: {len(documents)} chunks creados")
            return documents
            
        except Exception as e:
            logger.error(f"Error cargando DOCX {file_path}: {str(e)}")
            raise
    
    def load_txt(self, file_path: str) -> List[Document]:
        """Cargar documento TXT"""
        try:
            logger.info(f"Cargando TXT: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            metadata = {
                "source": file_path,
                "file_type": "txt"
            }
            
            documents = self.text_splitter.create_documents(
                texts=[text],
                metadatas=[metadata]
            )
            
            logger.info(f"TXT procesado: {len(documents)} chunks creados")
            return documents
            
        except Exception as e:
            logger.error(f"Error cargando TXT {file_path}: {str(e)}")
            raise
    
    def load_document(self, file_path: str) -> List[Document]:
        """Cargar documento según extensión"""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        extension = path.suffix.lower()
        
        if extension == '.pdf':
            return self.load_pdf(file_path)
        elif extension == '.docx':
            return self.load_docx(file_path)
        elif extension == '.txt':
            return self.load_txt(file_path)
        else:
            raise ValueError(f"Formato no soportado: {extension}")
    
    def load_directory(
        self, 
        directory_path: str,
        file_types: Optional[List[str]] = None
    ) -> List[Document]:
        """Cargar todos los documentos de un directorio"""
        if file_types is None:
            file_types = ['.pdf', '.docx', '.txt']
        
        directory = Path(directory_path)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directorio no encontrado: {directory_path}")
        
        all_documents = []
        
        for file_type in file_types:
            files = list(directory.glob(f"*{file_type}"))
            logger.info(f"Encontrados {len(files)} archivos {file_type}")
            
            for file_path in files:
                try:
                    documents = self.load_document(str(file_path))
                    all_documents.extend(documents)
                except Exception as e:
                    logger.error(f"Error procesando {file_path}: {str(e)}")
                    continue
        
        logger.info(f"Total de chunks cargados: {len(all_documents)}")
        return all_documents