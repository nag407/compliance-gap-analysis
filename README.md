# AI-Augmented Compliance Gap Analysis

A retrieval-augmented (RAG) tool that assesses an organization's security policy
against a controls framework (NIST CSF 2.0), producing cited, confidence-scored
**met / partial / gap** verdicts — with a built-in honesty layer that abstains and
flags uncertain cases for human review rather than risking false assurance.

> **Design philosophy:** the LLM *drafts and flags*; it never issues a verdict of record.
> Trustworthiness comes from the architecture around the model — abstain gates, confidence
> scoring, and human-in-the-loop — not from the model being assumed correct.

---

## Why this design

In compliance, a **false "met"** (claiming a control is satisfied when it isn't) is far more
dangerous than a false "gap" — it produces *false assurance* that can hide a real exposure or
fail an audit. This tool is built around preventing that specific failure:

- **Grounded verdicts** — every judgment is made over retrieved policy evidence, not model memory.
- **Abstain gate** — if no evidence is relevant enough, the tool refuses to answer and flags for review.
- **Confidence + review flags** — partial verdicts, low confidence, and parse failures are all routed to a human.
- **Evaluation harness** — verdicts are measured against a golden set, tracking a **false-assurance (false-met) rate**, not just raw accuracy.

## Architecture

The system is built around three swappable components behind clean interfaces, so moving from
local development to production is a **configuration change, not a rewrite**:

| Role | Interface | Dev implementation | Production path |
|---|---|---|---|
| Embedding | `Embedder` | Ollama (`nomic-embed-text`) | AWS Bedrock / managed embedding API |
| Vector store | `VectorStore` | ChromaDB (local) | pgvector on Aurora / OpenSearch |
| Judgment | `Judge` | Ollama (`llama3.1:8b`) | Claude / Bedrock (with native JSON mode) |

Components are injected into the pipeline, which depends only on the interfaces — never on a
specific tool. A hybrid retrieval store (BM25 + dense, merged via Reciprocal Rank Fusion) is
included as a pluggable capability.

### Pipeline
`ingest → chunk (structure-based) → embed → store → retrieve → abstain-gate → judge → gap register (+ review flags)`

## Pipeline flow

    Policy document
         │  chunk on section boundaries
         ▼
    Embed chunks ──► Vector store (with project_id metadata for multi-tenancy)
                          │
    Control ──► embed ──► retrieve top-k ──► relevance gate ──┐
                                                              │ below threshold
                          ┌───────────────────────────────────┤
                          ▼ (passes)                           ▼
                     LLM judge                            abstain →
                (met/partial/gap                        needs_review
                 + confidence + citation)
                          │
                          ▼
                  Gap register  (+ ⚠ review flags on uncertain verdicts)

## Usage

```bash
# 1. Install deps (Python 3.12)
python -m venv venv && venv\Scripts\activate      # Windows
pip install -r requirements.txt

# 2. Requires Ollama running locally with:
ollama pull nomic-embed-text
ollama pull llama3.1:8b

# 3. Run an assessment (produces the gap register)
python main.py

# 4. Run the evaluation harness (scores against the golden set)
python evaluate.py
```

## Evaluation

`evaluate.py` scores the tool against a hand-labelled golden set (`expected_verdicts.md`),
reporting exact-match accuracy and — the metric that matters most — the **false-met count**.

On the synthetic test policy, the tool achieves **0 false-assurance verdicts**, with all errors
biased toward caution (flagging for review rather than over-claiming compliance) — the intended
behavior. Accuracy tuning (thresholds, reranking, model choice) is deliberately deferred to real
data with a real golden set, since tuning against synthetic data optimizes for the wrong distribution.

## Status & roadmap

Built as a **production-architected** prototype: the design, interfaces, safety layers, and eval
harness are in place; deployment, scale-hardening, and accuracy tuning are the production next steps.

- [x] Swappable Embedder / VectorStore / Judge interfaces
- [x] Structure-based chunking, abstain gate, confidence + review flags
- [x] Retry loop + fail-safe LLM output handling
- [x] Multi-tenancy (project_id metadata) + normalized similarity
- [x] Hybrid retrieval (BM25 + dense, RRF) as a pluggable capability
- [x] Evaluation harness with false-met tracking
- [ ] Config-driven component selection
- [ ] Reranker stage (designed-in; highest-ROI on large corpora)
- [ ] Production adapters (pgvector, Bedrock/Claude) + AWS deployment

## Note on data

The included `sample_policy.txt` is **synthetic** (fictional "Northwind Technologies"). No real
organizational data is used, consistent with the tool's own data-sensitivity principles.