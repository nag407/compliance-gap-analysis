from embedders import OllamaEmbedder
from vector_stores import ChromaStore
from judges import OllamaJudge

class GapAnalysisPipeline:
    def __init__(self, embedder, store, judge, relevance_threshold=0.35):
        # note: pipeline depends on INTERFACES, not specific tools —
        # these are passed in, so swapping components happens at construction
        self.embedder = embedder
        self.store = store
        self.judge = judge
        self.relevance_threshold = relevance_threshold

    def ingest(self, chunks: list[dict], project_id: str = "default"):
        """chunks: list of {'id', 'text'}. Embeds and stores each with tenancy metadata."""
        for c in chunks:
            vector = self.embedder.embed(c["text"])
            self.store.add(
                chunk_id=f"{project_id}::{c['id']}",
                text=c["text"],
                vector=vector,
                metadata={"project_id": project_id, "source": c.get("source", "policy")},
            )
        print(f"Ingested {len(chunks)} chunks for project '{project_id}'")

    def assess_control(self, control: dict, project_id: str = "default") -> dict:
        # 1. RETRIEVE
        qvec = self.embedder.embed(control["text"])
        results = self.store.query(qvec, top_k=1, where={"project_id": project_id},
                                   query_text=control["text"])

        if not results:
            return {"control": control["id"], "verdict": "no_evidence", "needs_review": True}

        top = results[0]

        # 2. ABSTAIN GATE
        if top["similarity"] < self.relevance_threshold:
            return {"control": control["id"], "verdict": "insufficient_evidence",
                    "similarity": top["similarity"], "needs_review": True}

        # 3. JUDGE
        verdict = self.judge.assess(control["text"], top["text"])
        verdict["control"] = control["id"]
        verdict["similarity"] = top["similarity"]
        verdict["needs_review"] = verdict.get("verdict") in ("partial", "parse_error") or \
                                  (verdict.get("confidence") or 0) < 0.5
        return verdict