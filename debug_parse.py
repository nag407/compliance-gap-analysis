import ollama

control = "Actions performed during an investigation are recorded, and the records' integrity and provenance are preserved"
evidence = "Suspected information security incidents must be reported promptly to the Security Operations team through the established reporting channel. Reported events are triaged, and confirmed incidents are managed through a defined incident-response process covering containment, eradication, and recovery."

prompt = f"""You are a rigorous, skeptical compliance auditor. Avoid FALSE ASSURANCE — never mark a control "met" unless the evidence SPECIFICALLY and EXPLICITLY satisfies it.
Respond with ONLY JSON: {{"verdict":"met"|"partial"|"gap","confidence":0.0-1.0,"reasoning":"..."}}
CONTROL: {control}
EVIDENCE: {evidence}"""

resp = ollama.chat(model="llama3.1:8b", messages=[{"role":"user","content":prompt}], options={"temperature":0})
raw = resp["message"]["content"]
print("=== RAW OUTPUT ===")
print(repr(raw))   # repr shows hidden characters, quotes, newlines exactly