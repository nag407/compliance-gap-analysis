import ollama
from interfaces import Embedder

class OllamaEmbedder(Embedder):
    def __init__(self, model: str = "nomic-embed-text"):
        self.model = model

    def embed(self, text: str) -> list[float]:
        return ollama.embeddings(model=self.model, prompt=text)["embedding"]