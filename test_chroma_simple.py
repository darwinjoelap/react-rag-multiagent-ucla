"""
Script de diagnóstico SIMPLIFICADO para verificar documentos indexados
Compatible con cualquier estructura de proyecto
"""

import sys
import os

# Agregar el directorio backend al path
backend_path = os.path.join(os.getcwd(), 'backend')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

import chromadb

def print_header(text):
    """Imprime un encabezado destacado"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)

def test_chromadb_direct():
    """Diagnóstico usando ChromaDB directamente"""
    print_header("🔬 DIAGNÓSTICO DE CHROMADB - MODO DIRECTO")
    
    try:
        # Conectar a ChromaDB
        print("\n⏳ Conectando a ChromaDB...")
        chroma_path = os.path.join(backend_path, "data", "chroma_db")
        
        if not os.path.exists(chroma_path):
            print(f"❌ Error: No existe el directorio {chroma_path}")
            print(f"   Asegúrate de que la base de datos esté inicializada")
            return
        
        client = chromadb.PersistentClient(path=chroma_path)
        print(f"✅ Conectado a ChromaDB: {chroma_path}")
        
        # Obtener colección
        collection_name = "ucla_documents"
        try:
            collection = client.get_collection(name=collection_name)
            print(f"✅ Colección '{collection_name}' encontrada")
        except Exception as e:
            print(f"❌ Error: Colección '{collection_name}' no encontrada")
            print(f"   Colecciones disponibles: {client.list_collections()}")
            return
        
        # ============================================================================
        # 1. OBTENER TODOS LOS DOCUMENTOS
        # ============================================================================
        print_header("📊 DISTRIBUCIÓN DE DOCUMENTOS POR FUENTE")
        
        all_docs = collection.get()
        total_docs = len(all_docs['ids'])
        
        print(f"\n📈 Total de documentos: {total_docs}")
        
        if total_docs == 0:
            print("⚠️ La colección está vacía")
            return
        
        # Contar por fuente
        sources = {}
        for metadata in all_docs['metadatas']:
            source = metadata.get('source', 'unknown')
            filename = source.split('/')[-1] if '/' in source else source
            filename = filename.split('\\')[-1]  # Windows compatibility
            sources[filename] = sources.get(filename, 0) + 1
        
        print("\n📁 Archivos indexados:")
        for filename, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * min(count // 10, 50)
            print(f"  • {filename:60s} {count:4d} chunks {bar}")
        
        print(f"\n📈 Resumen:")
        print(f"  • Total de fuentes: {len(sources)}")
        print(f"  • Total de chunks: {sum(sources.values())}")
        
        # ============================================================================
        # 2. VERIFICACIÓN DE ARCHIVOS NUEVOS
        # ============================================================================
        print_header("📋 VERIFICACIÓN DE ARCHIVOS NUEVOS")
        
        new_files = [
            "Redes Neuronales Artificiales.pdf",
            "Búsqueda Heurística en Inteligencia Artificial",
            "Machine Learning.pdf",
            "Módulo2_Agentes.pdf",
            "Introducción al Machine Learning.pdf",
            "Introducción al ML.pdf",
            "Machine Learning_Part1 (Spanish).pdf",
        ]
        
        print("\n✓ Verificando archivos nuevos:")
        found_count = 0
        for new_file in new_files:
            found = any(new_file.lower() in filename.lower() for filename in sources.keys())
            count = next((cnt for filename, cnt in sources.items() if new_file.lower() in filename.lower()), 0)
            
            if found:
                print(f"  ✅ {new_file:60s} {count:4d} chunks")
                found_count += 1
            else:
                print(f"  ❌ {new_file:60s} NO ENCONTRADO")
        
        print(f"\n📊 Archivos nuevos encontrados: {found_count}/{len(new_files)}")
        
        # ============================================================================
        # 3. MUESTRA DE DOCUMENTOS
        # ============================================================================
        print_header("📄 MUESTRA DE DOCUMENTOS INDEXADOS")
        
        print("\n🔍 Primeros 5 documentos:")
        for i in range(min(5, total_docs)):
            doc_id = all_docs['ids'][i]
            metadata = all_docs['metadatas'][i]
            document = all_docs['documents'][i]
            
            source = metadata.get('source', 'unknown')
            filename = source.split('/')[-1] if '/' in source else source
            filename = filename.split('\\')[-1]
            chunk_id = metadata.get('chunk_id', 'N/A')
            
            preview = document[:100] + "..." if len(document) > 100 else document
            
            print(f"\n  [{i+1}] {filename} (chunk {chunk_id})")
            print(f"      ID: {doc_id}")
            print(f"      Preview: {preview}")
        
        # ============================================================================
        # 4. RESUMEN FINAL
        # ============================================================================
        print_header("✅ RESUMEN DEL DIAGNÓSTICO")
        
        print(f"""
📊 Estadísticas Generales:
  • Total de chunks indexados: {total_docs}
  • Total de fuentes únicas: {len(sources)}
  • Archivos nuevos encontrados: {found_count}/{len(new_files)}

🎯 Estado del Sistema:
  • ChromaDB: {'✅ Operativo' if total_docs > 0 else '❌ Vacío'}
  • Archivos Nuevos: {'✅ Indexados correctamente' if found_count >= len(new_files) * 0.8 else '⚠️ Algunos faltantes'}
  • Colección: {collection_name}

💡 Recomendaciones:
  • {'✅ Sistema listo para demo' if found_count >= 5 and total_docs > 600 else '⚠️ Verificar indexación'}
  • {'✅ Suficientes documentos indexados' if total_docs > 500 else '⚠️ Considerar más documentos'}
        """)
        
        print("=" * 80)
        print("🎉 Diagnóstico completado".center(80))
        print("=" * 80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error durante el diagnóstico: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_chromadb_direct()
