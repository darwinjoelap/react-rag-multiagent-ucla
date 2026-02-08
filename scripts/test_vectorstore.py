import sys
sys.path.insert(0, '../backend')

from app.services.vector_store import VectorStoreService
from app.services.document_loader import DocumentLoader
from pathlib import Path

def main():
    print("🚀 Probando Vector Store con ChromaDB\n")
    
    # 1. Inicializar servicios
    print("1️⃣ Inicializando servicios...")
    vector_store = VectorStoreService()
    doc_loader = DocumentLoader()
    
    # 2. Cargar documentos
    print("\n2️⃣ Cargando documentos desde data/raw...")
    documents = doc_loader.load_directory("../data/raw")
    print(f"   Documentos cargados: {len(documents)} chunks")
    
    if len(documents) == 0:
        print("   ⚠️  No hay documentos en data/raw/")
        print("   Por favor agrega algunos PDFs y vuelve a ejecutar.")
        return
    
    # 3. Agregar a ChromaDB
    print("\n3️⃣ Agregando documentos al vector store...")
    result = vector_store.add_documents(documents)
    print(f"   ✅ Agregados: {result['added']} documentos")
    print(f"   📊 Total en colección: {result['total']}")
    
    # 4. Estadísticas
    print("\n4️⃣ Estadísticas del vector store:")
    stats = vector_store.get_stats()
    for key, value in stats.items():
        print(f"   • {key}: {value}")
    
    # 5. Probar búsqueda
    print("\n5️⃣ Probando búsqueda...")
    queries = [
        "¿Qué es inteligencia artificial?",
        "machine learning",
        "neural networks"
    ]
    
    for query in queries:
        print(f"\n   📝 Query: '{query}'")
        results = vector_store.search(query, n_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n      Resultado {i}:")
            print(f"      • Similitud: {result['similarity']:.4f}")
            print(f"      • Fuente: {result['metadata'].get('source', 'N/A')}")
            print(f"      • Preview: {result['document'][:150]}...")
    
    print("\n\n✅ Prueba completada exitosamente!")

if __name__ == "__main__":
    main()