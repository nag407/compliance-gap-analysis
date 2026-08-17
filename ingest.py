import ollama
import numpy as np
import json
import re
# ---------- CHUNKING ----------
def chunk_text(text):
    """Split on section boundaries (numbered headings) so each chunk is one coherent section."""
    # (?:...) = non-capturing group, so re.split won't inject None/matches into the output
    parts = re.split(r'(?=\n?\s*\d+(?:\.\d+)?\.?\s+[A-Z])', text)
    chunks = []
    for p in parts:
        if not p:                     # skip None or empty pieces, just in case
            continue
        p = " ".join(p.split())       # normalize whitespace
        if len(p) > 30:               # skip tiny fragments
            chunks.append(p)
    return chunks

# ---------- EMBEDDING ----------
def embed(text):
    return ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]

# ---------- INGEST ----------
def ingest(policy_path):
    with open(policy_path, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = chunk_text(text)
    print(f"Split policy into {len(chunks)} chunks\n")

    # embed each chunk and build our store (list of dicts, for now)
    store = []
    for idx, chunk in enumerate(chunks):
        vector = embed(chunk)
        store.append({"id": idx, "text": chunk, "vector": vector})
        print(f"  chunk {idx}: {chunk[:70]}...")

    # save to disk so we don't re-embed every run
    with open("policy_store.json", "w", encoding="utf-8") as f:
        json.dump(store, f)
    print(f"\nStored {len(store)} embedded chunks to policy_store.json")

if __name__ == "__main__":
    ingest("sample_policy.txt")