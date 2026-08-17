import ollama, numpy as np, json

def embed(text):
    return np.array(ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"])

def similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# load the embedded policy chunks
with open("policy_store.json", "r", encoding="utf-8") as f:
    store = json.load(f)
for c in store:
    c["vector"] = np.array(c["vector"])

RELEVANCE_THRESHOLD = 0.55

controls = [
    {"id": "PR.AA-05", "text": "Access permissions, entitlements, and authorizations are defined in a policy, managed, enforced, and reviewed, and incorporate the principles of least privilege and separation of duties"},
    {"id": "PR.AT-01", "text": "Personnel are provided with awareness and training so that they possess the knowledge and skills to perform general tasks with cybersecurity risks in mind"},
    {"id": "PR.DS-11", "text": "Backups of data are created, protected, maintained, and tested"},
    {"id": "PR.PS-05", "text": "Installation and execution of unauthorized software are prevented"},
    {"id": "DE.CM-09", "text": "Computing hardware and software, runtime environments, and their data are monitored to find potentially adverse events"},
    {"id": "ID.AM-05", "text": "Assets are prioritized based on classification, criticality, resources, and impact on the mission"},
    {"id": "ID.RA-05", "text": "Threats, vulnerabilities, likelihoods, and impacts are used to understand inherent risk and inform risk response prioritization"},
    {"id": "DE.AE-08", "text": "Incidents are declared when adverse events meet the defined incident criteria"},
    {"id": "RS.CO-03", "text": "Information is shared with designated internal and external stakeholders"},
    {"id": "RC.RP-01", "text": "The recovery portion of the incident response plan is executed once initiated from the incident response process"},
    {"id": "ID.IM-03", "text": "Improvements are identified from execution of operational processes, procedures, and activities"},
    {"id": "GV.OV-03", "text": "Organizational cybersecurity risk management performance is evaluated and reviewed for adjustments needed"},
    {"id": "PR.IR-04", "text": "Adequate resource capacity to ensure availability is maintained"},
    {"id": "RS.AN-06", "text": "Actions performed during an investigation are recorded, and the records' integrity and provenance are preserved"},
]

def assess(control):
    cv = embed(control["text"])
    ranked = sorted(store, key=lambda c: similarity(cv, c["vector"]), reverse=True)
    top = ranked[0]
    top_score = similarity(cv, top["vector"])

    if top_score < RELEVANCE_THRESHOLD:
        return {"verdict": "insufficient_evidence", "confidence": None,
                "score": round(top_score, 3), "needs_review": True}

    context = top["text"]
    prompt = f"""You are a rigorous, skeptical compliance auditor. Your job is to avoid FALSE ASSURANCE — never mark a control "met" unless the evidence SPECIFICALLY and EXPLICITLY satisfies it.

Rules:
- "met": the evidence explicitly and specifically satisfies EVERY part of the control.
- "partial": the evidence addresses the general topic but lacks specific detail, or satisfies only part of the control.
- "gap": the evidence does not meaningfully address the specific requirement of the control, even if it mentions the general topic area.

Be strict: mentioning a topic is NOT the same as satisfying the control. If the evidence is only topically related but does not address the specific requirement, that is a "gap", not "met".

Respond with ONLY JSON: {{"verdict":"met"|"partial"|"gap","confidence":0.0-1.0,"reasoning":"cite the specific evidence text, or state what is missing"}}

CONTROL: {control['text']}

EVIDENCE: {context}"""
    resp = ollama.chat(model="llama3.1:8b", messages=[{"role":"user","content":prompt}],
                       options={"temperature": 0})
    try:
        out = json.loads(resp["message"]["content"])
    except:
        out = {"verdict": "parse_error", "raw": resp["message"]["content"]}
    out["score"] = round(top_score, 3)
    return out

print(f"{'CONTROL':<10} {'VERDICT':<22} {'CONF':<6} {'SCORE'}")
print("-" * 55)
for ctrl in controls:
    r = assess(ctrl)
    conf = r.get("confidence")
    conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "-"
    print(f"{ctrl['id']:<10} {r.get('verdict',''):<22} {conf_str:<6} {r.get('score','')}")