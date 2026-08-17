import chromadb
from interfaces import VectorStore

class ChromaStore(VectorStore):
    def __init__(self, collection_name: str = "evidence", path: str = ".chroma"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"}   # standardize on cosine
        )

    def add(self, chunk_id: str, text: str, vector: list[float], metadata: dict) -> None:
        self.collection.upsert(
            ids=[chunk_id],
            documents=[text],
            embeddings=[vector],
            metadatas=[metadata],
        )

    def query(self, vector, top_k, where=None, query_text=None): 
        res = self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=where,                       # multi-tenancy filter
        )
        # normalize Chroma's response into plain dicts (so the interface stays tool-agnostic)
        out = []
        for i in range(len(res["ids"][0])):
            cosine_distance = res["distances"][0][i]     # Chroma cosine distance: 0..2
            similarity = 1 - cosine_distance             # -> 0..1, higher = better
            out.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "similarity": round(similarity, 4),      # the ONLY score callers see
            })
        return out