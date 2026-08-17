from embedders import OllamaEmbedder
from vector_stores import ChromaStore

embedder = OllamaEmbedder()
store = ChromaStore(collection_name="test")

store.add("c1", "MFA is required for admin accounts.",
          embedder.embed("MFA is required for admin accounts."), {"source": "test"})
store.add("c2", "The cafeteria serves lunch at noon.",
          embedder.embed("The cafeteria serves lunch at noon."), {"source": "test"})

results = store.query(embedder.embed("Is multi-factor authentication enforced?"), top_k=2)
print("Query results (most similar first):")
for r in results:
    print(f"  similarity={r['similarity']:.3f}  {r['text']}")