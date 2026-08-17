import re
from embedders import OllamaEmbedder
from vector_stores import ChromaStore
from judges import OllamaJudge
from pipeline import GapAnalysisPipeline

def chunk_policy(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    parts = re.split(r'(?=\n?\s*\d+(?:\.\d+)?\.?\s+[A-Z])', text)
    chunks, idx = [], 0
    for p in parts:
        if not p: continue
        p = " ".join(p.split())
        if len(p) > 30:
            chunks.append({"id": f"chunk-{idx}", "text": p})
            idx += 1
    return chunks

controls = [
    {"id": "GV.OV-03", "text": "Organizational cybersecurity risk management performance is evaluated and reviewed for adjustments needed"},
    {"id": "ID.AM-05", "text": "Assets are prioritized based on classification, criticality, resources, and impact on the mission"},
    {"id": "ID.RA-05", "text": "Threats, vulnerabilities, likelihoods, and impacts are used to understand inherent risk and inform risk response prioritization"},
    {"id": "ID.IM-03", "text": "Improvements are identified from execution of operational processes, procedures, and activities"},
    {"id": "PR.AA-05", "text": "Access permissions, entitlements, and authorizations are defined in a policy, managed, enforced, and reviewed, and incorporate the principles of least privilege and separation of duties"},
    {"id": "PR.AT-01", "text": "Personnel are provided with awareness and training so that they possess the knowledge and skills to perform general tasks with cybersecurity risks in mind"},
    {"id": "PR.DS-11", "text": "Backups of data are created, protected, maintained, and tested"},
    {"id": "PR.PS-05", "text": "Installation and execution of unauthorized software are prevented"},
    {"id": "PR.IR-04", "text": "Adequate resource capacity to ensure availability is maintained"},
    {"id": "DE.CM-09", "text": "Computing hardware and software, runtime environments, and their data are monitored to find potentially adverse events"},
    {"id": "DE.AE-08", "text": "Incidents are declared when adverse events meet the defined incident criteria"},
    {"id": "RS.AN-06", "text": "Actions performed during an investigation are recorded, and the records' integrity and provenance are preserved"},
    {"id": "RS.CO-03", "text": "Information is shared with designated internal and external stakeholders"},
    {"id": "RC.RP-01", "text": "The recovery portion of the incident response plan is executed once initiated from the incident response process"},
]

if __name__ == "__main__":
    # THE SWAP POINT: change these three lines to go to production later
    embedder = OllamaEmbedder()
    store = ChromaStore(collection_name="gap_analysis")
    judge = OllamaJudge(model="llama3.1:8b")

    pipeline = GapAnalysisPipeline(embedder, store, judge)

    chunks = chunk_policy("sample_policy.txt")
    pipeline.ingest(chunks, project_id="northwind")

    print(f"\n{'CONTROL':<10} {'VERDICT':<22} {'CONF':<6} {'SIM':<6} REVIEW")
    print("-" * 60)
    for ctrl in controls:
        r = pipeline.assess_control(ctrl, project_id="northwind")
        conf = r.get("confidence")
        conf_str = f"{conf:.2f}" if isinstance(conf, (int, float)) else "-"
        sim = r.get("similarity", 0)
        review = "⚠ REVIEW" if r.get("needs_review") else ""
        print(f"{r['control']:<10} {r.get('verdict',''):<22} {conf_str:<6} {sim:<6.3f} {review}")