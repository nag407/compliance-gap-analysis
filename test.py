import ollama
import numpy as np

def embed(text):
    return np.array(ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"])

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

evidence = [
    "All administrative accounts must authenticate using multi-factor authentication.",
    "Backups are performed nightly and stored offsite for 90 days.",
    "Server room access requires badge entry and is logged.",
    "Passwords must be at least 12 characters and rotated every 90 days.",
]
evidence_vectors = [embed(chunk) for chunk in evidence]

RELEVANCE_THRESHOLD = 0.55   # tune this against real data later

def assess(control):
    cv = embed(control)
    ranked = sorted(zip((similarity(cv, v) for v in evidence_vectors), evidence), reverse=True)
    top_score, top_chunk = ranked[0]

    # THE ABSTAIN GATE
    if top_score < RELEVANCE_THRESHOLD:
        return {
            "verdict": "insufficient_evidence",
            "needs_human_review": True,
            "reason": f"No evidence relevant enough (best match scored {top_score:.3f}, below {RELEVANCE_THRESHOLD})"
        }

    # Passed the gate -> judge using the top chunk
    prompt = f"""You are a compliance assessor. Using ONLY the evidence below, judge the control.
Respond with ONLY a JSON object:
{{"verdict": "met" or "partial" or "gap", "confidence": 0.0 to 1.0, "reasoning": "one sentence", "citation": "the exact evidence sentence you relied on"}}

CONTROL: {control}
EVIDENCE:
- {top_chunk}"""
    resp = ollama.chat(model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0})
    return resp["message"]["content"]

# TEST 1: a control the evidence CAN answer (should pass the gate -> "met")
print("--- TEST 1: MFA (evidence exists) ---")
print(assess("Is multi-factor authentication enforced for administrators?"))

# TEST 2: a control the evidence CANNOT answer (should ABSTAIN)
print("\n--- TEST 2: incident response (no evidence exists) ---")
print(assess("Does the organization have a documented incident response plan?"))