from embedders import OllamaEmbedder   # adjust import to your file location

embedder = OllamaEmbedder()
vec = embedder.embed("Multi-factor authentication is required.")
print(f"Got a vector of length {len(vec)} via the Embedder interface")