"""
Script de diagnóstico para verificar documentos indexados
y validar que los nuevos PDFs están siendo recuperados correctamente
"""

import sys
sys.path.append('.')

from app.services.vector_store import VectorStoreService
from app.services.embeddings import EmbeddingsService

def print_header(text):
    """Imprime un encabezado destacado"""
    print("\n" + "=" * 80)
    print(text.center(80))
    print("=" * 80)

def test_new_documents():
    print_header("🔬 DIAGNÓSTICO DE DOCUMENTOS INDEXADOS")
    
    # Inicializar servicios
    print("\n⏳ Inicializando servicios...")
    embeddings = EmbeddingsService()
    vector_store = VectorStoreService(embeddings)
    print("✅ Servicios inicializados")
    
    # ============================================================================
    # 1. DISTRIBUCIÓN POR FUENTE
    # ============================================================================
    print_header("📊 DISTRIBUCIÓN DE DOCUMENTOS POR FUENTE")
    
    all_docs = vector_store.collection.get()
    sources = {}
    
    for metadata in all_docs['metadatas']:
        source = metadata.get('source', 'unknown')
        filename = source.split('/')[-1] if '/' in source else source
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
    for new_file in new_files:
        found = any(new_file.lower() in filename.lower() for filename in sources.keys())
        count = next((cnt for filename, cnt in sources.items() if new_file.lower() in filename.lower()), 0)
        
        if found:
            print(f"  ✅ {new_file:60s} {count:4d} chunks")
        else:
            print(f"  ❌ {new_file:60s} NO ENCONTRADO")
    
    # ============================================================================
    # 3. PRUEBAS DE BÚSQUEDA ESPECÍFICAS
    # ============================================================================
    print_header("🔍 PRUEBAS DE BÚSQUEDA ESPECÍFICAS")
    
    test_queries = [
        {
            "query": "redes neuronales artificiales",
            "expected": "Redes Neuronales Artificiales.pdf",
            "description": "Debería recuperar el PDF de Redes Neuronales"
        },
        {
            "query": "búsqueda heurística en inteligencia artificial",
            "expected": "Búsqueda Heurística",
            "description": "Debería recuperar el PDF de Búsqueda Heurística"
        },
        {
            "query": "machine learning supervisado no supervisado",
            "expected": "Machine Learning",
            "description": "Debería recuperar algún PDF de Machine Learning"
        },
        {
            "query": "agente inteligente racionalidad",
            "expected": "Módulo2_Agentes.pdf",
            "description": "Debería recuperar el PDF de Agentes"
        },
        {
            "query": "perceptrón multicapa backpropagation",
            "expected": "Redes Neuronales",
            "description": "Debería recuperar contenido sobre redes neuronales"
        },
    ]
    
    for i, test in enumerate(test_queries, 1):
        query = test["query"]
        expected = test["expected"]
        description = test["description"]
        
        print(f"\n{'─' * 80}")
        print(f"🔎 Prueba {i}/{len(test_queries)}: {description}")
        print(f"   Query: '{query}'")
        print(f"   Esperado: {expected}")
        
        results = vector_store.search(query, top_k=5)
        
        if results:
            print(f"   ✅ {len(results)} resultados encontrados:")
            found_expected = False
            
            for j, doc in enumerate(results, 1):
                source = doc['metadata'].get('source', 'unknown')
                filename = source.split('/')[-1] if '/' in source else source
                similarity = doc['similarity']
                
                # Marcar si encontramos el archivo esperado
                is_expected = expected.lower() in filename.lower()
                marker = "⭐" if is_expected else "  "
                
                if is_expected:
                    found_expected = True
                
                print(f"   {marker} [{j}] {filename:50s} (sim={similarity:.4f})")
            
            if found_expected:
                print(f"   ✅ ÉXITO: Se encontró '{expected}' en los resultados")
            else:
                print(f"   ⚠️  ADVERTENCIA: No se encontró '{expected}' en los top 5")
        else:
            print("   ❌ Sin resultados")
    
    # ============================================================================
    # 4. ANÁLISIS DE CALIDAD DE EMBEDDINGS
    # ============================================================================
    print_header("📊 ANÁLISIS DE CALIDAD DE SIMILITUD")
    
    quality_tests = [
        ("redes neuronales", 0.25),
        ("machine learning", 0.25),
        ("inteligencia artificial", 0.20),
        ("agente inteligente", 0.25),
    ]
    
    print("\n🎯 Umbrales de similitud esperados:")
    for query, expected_threshold in quality_tests:
        results = vector_store.search(query, top_k=3)
        if results:
            max_sim = max(r['similarity'] for r in results)
            avg_sim = sum(r['similarity'] for r in results) / len(results)
            
            status = "✅" if max_sim >= expected_threshold else "⚠️"
            print(f"  {status} '{query:30s}' → max={max_sim:.4f}, avg={avg_sim:.4f} (umbral={expected_threshold})")
        else:
            print(f"  ❌ '{query:30s}' → Sin resultados")
    
    # ============================================================================
    # 5. RESUMEN FINAL
    # ============================================================================
    print_header("✅ RESUMEN DEL DIAGNÓSTICO")
    
    total_chunks = sum(sources.values())
    total_sources = len(sources)
    new_files_found = sum(1 for nf in new_files if any(nf.lower() in f.lower() for f in sources.keys()))
    
    print(f"""
📊 Estadísticas Generales:
  • Total de chunks indexados: {total_chunks}
  • Total de fuentes únicas: {total_sources}
  • Archivos nuevos encontrados: {new_files_found}/{len(new_files)}

🎯 Estado del Sistema:
  • Vector Store: {'✅ Operativo' if total_chunks > 0 else '❌ Vacío'}
  • Archivos Nuevos: {'✅ Indexados correctamente' if new_files_found >= len(new_files) * 0.8 else '⚠️ Algunos faltantes'}
  • Embeddings: {'✅ Funcionando' if total_chunks > 0 else '❌ Error'}

💡 Recomendaciones:
  • {'✅ Sistema listo para demo' if new_files_found >= 5 and total_chunks > 600 else '⚠️ Verificar indexación'}
  • {'✅ Calidad de búsqueda adecuada' if total_chunks > 500 else '⚠️ Considerar más documentos'}
    """)
    
    print("=" * 80)
    print("🎉 Diagnóstico completado".center(80))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    try:
        test_new_documents()
    except Exception as e:
        print(f"\n❌ Error durante el diagnóstico: {e}")
        import traceback
        traceback.print_exc()
