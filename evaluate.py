from embedders import OllamaEmbedder
from vector_stores import ChromaStore
from judges import OllamaJudge
from pipeline import GapAnalysisPipeline
from main import chunk_policy, controls        # reuse what you already have
from hybrid_store import HybridStore

# THE GOLDEN SET: your expected verdicts (from expected_verdicts.md)
EXPECTED = {
    "PR.AA-05": "met",   "PR.AT-01": "met",   "PR.DS-11": "met",
    "PR.PS-05": "met",   "DE.CM-09": "met",
    "ID.AM-05": "partial", "ID.RA-05": "partial", "DE.AE-08": "partial",
    "RS.CO-03": "partial", "RC.RP-01": "partial", "ID.IM-03": "partial",
    "GV.OV-03": "gap",   "PR.IR-04": "gap",   "RS.AN-06": "gap",
}

def run_eval():
    embedder = OllamaEmbedder()
    store = HybridStore(collection_name="gap_hybrid") 
    judge = OllamaJudge(model="llama3.1:8b")
    pipeline = GapAnalysisPipeline(embedder, store, judge)

    pipeline.ingest(chunk_policy("sample_policy.txt"), project_id="eval")

    correct = 0
    false_met = 0          # the DANGEROUS error: said "met" when it wasn't
    total = len(controls)

    print(f"{'CONTROL':<10} {'EXPECTED':<10} {'GOT':<14} {'MATCH':<7} {'FLAG'}")
    print("-" * 55)
    for ctrl in controls:
        r = pipeline.assess_control(ctrl, project_id="eval")
        got = r.get("verdict", "error")
        expected = EXPECTED.get(ctrl["id"], "?")
        match = (got == expected)
        if match:
            correct += 1
        # false assurance = said "met" but expected partial or gap
        if got == "met" and expected in ("partial", "gap"):
            false_met += 1
        flag = "❌ FALSE-MET" if (got == "met" and expected != "met") else ("✓" if match else "✗")
        print(f"{ctrl['id']:<10} {expected:<10} {got:<14} {'YES' if match else 'no':<7} {flag}")

    print("-" * 55)
    print(f"\nExact-match accuracy: {correct}/{total} = {correct/total:.0%}")
    print(f"FALSE-MET count (false assurance): {false_met}   <-- the metric that matters most")

if __name__ == "__main__":
    run_eval()