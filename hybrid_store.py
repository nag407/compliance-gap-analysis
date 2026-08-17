import chromadb
import re
from rank_bm25 import BM25Okapi
from interfaces import VectorStore

def _tokenize(s):
    return re.findall(r"[a-z0-9]+", s.lower())

class HybridStore(VectorStore):
    """Dense (Chroma cosine) + sparse (BM25) retrieval, merged with Reciprocal Rank Fusion."""
    def __init__(self, collection_name="evidence_hybrid", path=".chroma", rrf_k=60):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(
            collection_name, metadata={"hnsw:space": "cosine"})
        self.rrf_k = rrf_k
        # in-memory corpus for BM25
        self._ids, self._tokens = [], []
        self._texts, self._metas = {}, {}

    def add(self, chunk_id, text, vector, metadata):
        self.collection.upsert(ids=[chunk_id], documents=[text],
                               embeddings=[vector], metadatas=[metadata])
        if chunk_id not in self._texts:
            self._ids.append(chunk_id)
            self._tokens.append(_tokenize(text))
        self._texts[chunk_id] = text
        self._metas[chunk_id] = metadata

    def _match(self, meta, where):
        return True if not where else all(meta.get(k) == v for k, v in where.items())

    def query(self, vector, top_k, where=None, query_text=None):
        # ---- DENSE: cosine similarity for candidates (Chroma applies the where filter) ----
        n = max(top_k, len(self._ids))
        dense = self.collection.query(query_embeddings=[vector], n_results=n, where=where)
        dense_sim, dense_rank = {}, []
        for i in range(len(dense["ids"][0])):
            cid = dense["ids"][0][i]
            dense_sim[cid] = 1 - dense["distances"][0][i]
            dense_rank.append(cid)                       # Chroma returns best-first

        # ---- SPARSE: BM25 over the in-memory corpus, respecting the where filter ----
        sparse_rank = []
        if query_text and self._tokens:
            bm25 = BM25Okapi(self._tokens)
            scores = bm25.get_scores(_tokenize(query_text))
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            sparse_rank = [self._ids[i] for i in order
                           if self._match(self._metas[self._ids[i]], where)]

        # ---- MERGE with Reciprocal Rank Fusion ----
        rrf = {}
        for rank_list in (dense_rank, sparse_rank):
            for rank, cid in enumerate(rank_list):
                rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        fused = sorted(rrf, key=lambda c: rrf[c], reverse=True)[:top_k]

        # ---- normalized output; report DENSE cosine as 'similarity' so the gate stays meaningful
        return [{
            "id": cid,
            "text": self._texts.get(cid, ""),
            "metadata": self._metas.get(cid, {}),
            "similarity": round(dense_sim.get(cid, 0.0), 4),
        } for cid in fused]