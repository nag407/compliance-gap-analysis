import ollama
import json
from interfaces import Judge

class OllamaJudge(Judge):
    def __init__(self, model: str = "llama3.1:8b"):
        self.model = model

    def assess(self, control_text: str, evidence: str) -> dict:
        prompt = f"""You are a rigorous, skeptical compliance auditor. Avoid FALSE ASSURANCE — never mark a control "met" unless the evidence SPECIFICALLY and EXPLICITLY satisfies it.
Rules:
- "met": evidence explicitly and specifically satisfies EVERY part of the control.
- "partial": evidence addresses the topic but lacks specific detail, or satisfies only part.
- "gap": evidence does not meaningfully address the specific requirement, even if it mentions the general topic area.
Mentioning a topic is NOT satisfying the control. Be strict.
Respond with ONLY JSON: {{"verdict":"met"|"partial"|"gap","confidence":0.0-1.0,"reasoning":"cite the specific evidence, or state what is missing"}}
CONTROL: {control_text}
EVIDENCE: {evidence}"""

        last_raw = ""
        for attempt in range(3):                       # up to 3 tries
            resp = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0},
            )
            last_raw = resp["message"]["content"].strip()
            start, end = last_raw.find("{"), last_raw.rfind("}")
            if start != -1 and end != -1:
                try:
                    return json.loads(last_raw[start:end + 1])
                except json.JSONDecodeError:
                    continue                            # malformed — retry
        # all attempts failed → fail safe, flag for human
        return {"verdict": "parse_error", "confidence": None,
                "reasoning": last_raw[:200], "needs_review": True}