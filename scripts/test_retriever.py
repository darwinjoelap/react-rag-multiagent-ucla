import sys
sys.path.insert(0, '../backend')

from app.services.retriever import RetrieverService
from app.services.vector_store import VectorStoreService

def print_separator(title=""):
    print("\n" + "="*70)
    if title:
        print(f"  {title}")
        print("="*70)

def main():
    print("🔍 Probando Retriever Service\n")
    
    # Inicializar
    print("Inicializando retriever...")
    retriever = RetrieverService(top_k=5, similarity_threshold=0.6)
    
    # Estadísticas
    print_separator("ESTADÍSTICAS DEL SISTEMA")
    stats = retriever.get_stats()
    for key, value in stats.items():
        print(f"  • {key}: {value}")
    
    # Consultas de prueba
    test_queries = [
        "¿Qué es inteligencia artificial?",
        "machine learning algorithms",
        "neural networks and deep learning",
        "natural language processing",
        "computer vision applications"
    ]
    
    print_separator("PRUEBA 1: RECUPERACIÓN BÁSICA")
    for query in test_queries[:2]:
        print(f"\n📝 Query: '{query}'")
        results = retriever.retrieve(query, top_k=3)
        
        print(f"   Resultados: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"\n   [{i}] Similitud: {result['similarity']:.4f}")
            print(f"       Fuente: {result['metadata'].get('source', 'N/A')}")
            print(f"       Preview: {result['document'][:120]}...")
    
    print_separator("PRUEBA 2: ANÁLISIS DE COBERTURA")
    for query in test_queries:
        coverage = retriever.analyze_query_coverage(query, top_k=5)
        print(f"\n📊 Query: '{query}'")
        print(f"   • Cobertura: {coverage['coverage']}")
        print(f"   • Similitud promedio: {coverage['avg_similarity']:.4f}")
        print(f"   • Similitud máxima: {coverage['max_similarity']:.4f}")
        print(f"   • Fuentes encontradas: {coverage['sources_found']}")
    
    print_separator("PRUEBA 3: CONTEXTO PARA LLM")
    query = "Explica qué es machine learning"
    print(f"\n📝 Query: '{query}'")
    context = retriever.get_relevant_context(query, max_tokens=500)
    print(f"\n   Contexto generado ({len(context)} caracteres):")
    print(f"\n{context[:500]}...")
    
    print_separator("PRUEBA 4: RECUPERACIÓN DIVERSA")
    query = "artificial intelligence"
    print(f"\n📝 Query: '{query}'")
    
    # Normal
    normal = retriever.retrieve(query, top_k=5)
    print(f"\n   Recuperación normal: {len(normal)} docs")
    
    # Diversa
    diverse = retriever.retrieve_diverse(query, top_k=5, diversity_factor=0.7)
    print(f"   Recuperación diversa: {len(diverse)} docs")
    
    print_separator("PRUEBA 5: BATCH RETRIEVAL")
    batch_queries = test_queries[:3]
    batch_results = retriever.batch_retrieve(batch_queries, top_k=2)
    
    for query, results in batch_results.items():
        print(f"\n   '{query}': {len(results)} resultados")
    
    print_separator()
    print("\n✅ Todas las pruebas completadas exitosamente!\n")

if __name__ == "__main__":
    main()