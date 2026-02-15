"""
Script para indexar documentos en ChromaDB desde data/raw

Uso:
    python index_documents.py
    
Opciones:
    python index_documents.py --reindex  # Borra colección existente y reindexar
    python index_documents.py --append   # Agregar a documentos existentes
"""

import sys
import os
from pathlib import Path
import argparse

# Agregar backend al path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingService
from langchain_core.documents import Document
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def read_text_file(file_path: Path) -> str:
    """Leer archivo de texto con múltiples encodings"""
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except (UnicodeDecodeError, LookupError):
            continue
    
    logger.warning(f"⚠️ No se pudo leer {file_path.name} con ningún encoding")
    return ""


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Dividir texto en chunks con overlap
    
    Args:
        text: Texto completo
        chunk_size: Tamaño de cada chunk en caracteres
        overlap: Caracteres de overlap entre chunks
        
    Returns:
        Lista de chunks
    """
    if not text or len(text) < 100:
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        
        # Intentar cortar en final de oración
        if end < text_length:
            # Buscar punto, salto de línea, o espacio
            for delimiter in ['. ', '.\n', '\n\n', '\n', ' ']:
                last_delim = chunk.rfind(delimiter)
                if last_delim > chunk_size * 0.5:
                    chunk = chunk[:last_delim + len(delimiter)]
                    end = start + last_delim + len(delimiter)
                    break
        
        chunk = chunk.strip()
        if len(chunk) > 50:  # Solo chunks con contenido real
            chunks.append(chunk)
        
        start = end - overlap
    
    return chunks


def index_documents(
    data_dir: str = "data/raw",
    chunk_size: int = 1000,
    overlap: int = 200,
    reindex: bool = False
):
    """
    Indexar documentos desde data/raw
    
    Args:
        data_dir: Directorio con documentos
        chunk_size: Tamaño de chunks
        overlap: Overlap entre chunks
        reindex: Si True, borra colección existente
    """
    logger.info("=" * 80)
    logger.info("🚀 INDEXACIÓN DE DOCUMENTOS - SISTEMA RAG UCLA")
    logger.info("=" * 80)
    
    # Ruta del proyecto
    project_root = Path(__file__).parent
    documents_dir = project_root / data_dir
    
    if not documents_dir.exists():
        logger.error(f"❌ Directorio no existe: {documents_dir}")
        logger.info(f"💡 Crear con: mkdir {documents_dir}")
        return
    
    # Buscar archivos soportados
    supported_formats = {
        '.txt': 'Texto',
        '.md': 'Markdown',
        '.pdf': 'PDF (requiere pypdf)',
    }
    
    all_files = []
    for ext in supported_formats.keys():
        files = list(documents_dir.glob(f"**/*{ext}"))
        all_files.extend(files)
        if files:
            logger.info(f"📄 {len(files)} archivos {supported_formats[ext]} encontrados")
    
    if not all_files:
        logger.error(f"❌ No se encontraron archivos en: {documents_dir}")
        logger.info(f"💡 Formatos soportados: {', '.join(supported_formats.keys())}")
        return
    
    logger.info(f"\n📁 Directorio: {documents_dir}")
    logger.info(f"📚 Total de archivos: {len(all_files)}")
    logger.info(f"✂️ Configuración: chunks={chunk_size} chars, overlap={overlap} chars")
    
    # Inicializar vector store
    logger.info("\n🔧 Inicializando ChromaDB...")
    
    if reindex:
        logger.warning("⚠️ MODO REINDEX: Se borrará la colección existente")
        input("Presiona Enter para continuar o Ctrl+C para cancelar...")
        
        try:
            # Crear instancia temporal para borrar
            temp_vs = VectorStoreService()
            temp_vs.client.delete_collection(name=temp_vs.collection_name)
            logger.info("✅ Colección anterior eliminada")
        except Exception as e:
            logger.info(f"ℹ️ No había colección previa: {e}")
    
    # Crear/conectar a vector store
    vector_store = VectorStoreService()
    initial_count = vector_store.collection.count()
    logger.info(f"📊 Documentos actuales en BD: {initial_count}")
    
    # Procesar archivos
    logger.info("\n" + "=" * 80)
    logger.info("📚 PROCESANDO DOCUMENTOS")
    logger.info("=" * 80)
    
    all_documents = []
    all_metadatas = []
    files_processed = 0
    files_skipped = 0
    
    for i, file_path in enumerate(all_files, 1):
        logger.info(f"\n[{i}/{len(all_files)}] 📖 {file_path.name}")
        
        # Leer contenido según extensión
        content = ""
        
        if file_path.suffix.lower() == '.pdf':
            logger.info("  ⏳ Leyendo PDF...")
            try:
                import pypdf
                with open(file_path, 'rb') as f:
                    pdf = pypdf.PdfReader(f)
                    content = "\n".join([page.extract_text() for page in pdf.pages])
                logger.info(f"  ✅ PDF leído: {len(pdf.pages)} páginas")
            except ImportError:
                logger.error("  ❌ Instala pypdf: pip install pypdf --break-system-packages")
                files_skipped += 1
                continue
            except Exception as e:
                logger.error(f"  ❌ Error leyendo PDF: {e}")
                files_skipped += 1
                continue
        
        else:  # TXT, MD
            content = read_text_file(file_path)
        
        if not content:
            logger.warning("  ⚠️ Archivo vacío o no se pudo leer")
            files_skipped += 1
            continue
        
        logger.info(f"  📏 Tamaño: {len(content):,} caracteres")
        
        # Dividir en chunks
        chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)
        
        if not chunks:
            logger.warning("  ⚠️ No se generaron chunks válidos")
            files_skipped += 1
            continue
        
        logger.info(f"  ✂️ Chunks generados: {len(chunks)}")
        
        # Preparar para indexar
        for j, chunk in enumerate(chunks):
            all_documents.append(chunk)
            all_metadatas.append({
                "source": file_path.name,
                "chunk_index": j,
                "total_chunks": len(chunks),
                "file_type": file_path.suffix.lower(),
                "char_count": len(chunk)
            })
        
        files_processed += 1
    
    # Indexar todo
    if all_documents:
        total_chunks = len(all_documents)
        logger.info("\n" + "=" * 80)
        logger.info(f"💾 INDEXANDO {total_chunks} CHUNKS EN CHROMADB")
        logger.info("=" * 80)
        
        # Indexar en batches
        batch_size = 100
        batches = (total_chunks + batch_size - 1) // batch_size
        
        for i in range(0, total_chunks, batch_size):
            end_idx = min(i + batch_size, total_chunks)
            batch_num = (i // batch_size) + 1
            
            logger.info(f"⏳ Batch {batch_num}/{batches}: Indexando chunks {i+1}-{end_idx}...")
            
            # Convertir a objetos Document de LangChain
            batch_documents = [
                Document(
                    page_content=all_documents[j],
                    metadata=all_metadatas[j]
                )
                for j in range(i, end_idx)
            ]
            
            # Indexar batch
            vector_store.add_documents(documents=batch_documents)
            
            logger.info(f"  ✅ Completado")
        
        # Estadísticas finales
        final_count = vector_store.collection.count()
        
        # Obtener fuentes únicas
        sample = vector_store.collection.get(
            limit=min(1000, final_count),
            include=["metadatas"]
        )
        
        sources = {}
        for metadata in sample['metadatas']:
            source = metadata.get('source', 'unknown')
            sources[source] = sources.get(source, 0) + 1
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ INDEXACIÓN COMPLETADA EXITOSAMENTE")
        logger.info("=" * 80)
        logger.info(f"📊 Archivos procesados: {files_processed}")
        logger.info(f"⚠️ Archivos saltados: {files_skipped}")
        logger.info(f"📄 Chunks indexados: {total_chunks}")
        logger.info(f"💾 Total en base de datos: {final_count} documentos")
        logger.info(f"📁 Fuentes únicas: {len(sources)} archivos")
        logger.info(f"📈 Incremento: +{final_count - initial_count} documentos")
        
        # Mostrar fuentes
        logger.info("\n📚 Fuentes indexadas:")
        for source in sorted(sources.keys())[:20]:
            count = sources[source]
            logger.info(f"  • {source}: {count} chunks")
        
        if len(sources) > 20:
            logger.info(f"  ... y {len(sources) - 20} fuentes más")
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 ¡Listo! El sistema RAG ya puede usar los nuevos documentos.")
        logger.info("=" * 80)
    
    else:
        logger.error("❌ No se pudo procesar ningún documento")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexar documentos en ChromaDB")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Borrar colección existente y reindexar desde cero"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000,
        help="Tamaño de cada chunk en caracteres (default: 1000)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=200,
        help="Overlap entre chunks en caracteres (default: 200)"
    )
    
    args = parser.parse_args()
    
    try:
        index_documents(
            chunk_size=args.chunk_size,
            overlap=args.overlap,
            reindex=args.reindex
        )
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Indexación cancelada por el usuario")
    except Exception as e:
        logger.error(f"\n❌ Error fatal: {e}", exc_info=True)
