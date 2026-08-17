from abc import ABC, abstractmethod

class Embedder(ABC):
    """Role: turn text into a vector."""
    @abstractmethod
    def embed(self, text: str) -> list[float]:
        ...

class VectorStore(ABC):
    """Role: store chunks + vectors, and retrieve the most similar."""
    @abstractmethod
    def add(self, chunk_id: str, text: str, vector: list[float], metadata: dict) -> None:
        ...

    @abstractmethod
    def query(self, vector: list[float], top_k: int, where: dict | None = None,
              query_text: str | None = None) -> list[dict]:
        """Returns list of {'id','text','metadata','similarity'} (similarity 0-1, higher=better).
           query_text is the raw query, used by hybrid stores for keyword search;
           dense-only stores may ignore it."""
        ...

class Judge(ABC):
    """Role: read evidence, judge a control, return a structured verdict."""
    @abstractmethod
    def assess(self, control_text: str, evidence: str) -> dict:
        ...