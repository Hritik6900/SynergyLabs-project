# Problem 1 — Cost-Efficient RAG Application

A question-answering service over a document corpus, backed by a **low-cost local
vector store (ChromaDB)**, with honest evaluation of retrieval quality, answer
quality, latency, and cost.

- **Store:** ChromaDB persistent client (embedded, on-disk, no server process).
- **Embeddings:** OpenAI `text-embedding-3-small` (1536-dim) — or a deterministic
  local fallback so the whole pipeline runs **offline with zero spend**.
- **Generation:** OpenAI `gpt-4o-mini` **or** Anthropic `claude-haiku-4-5`
  (configurable via env).
- **Interface:** FastAPI HTTP endpoints **and** a CLI.

---

## Why ChromaDB (store justification)

The brief's premise: on a fully managed vector DB the bill scales with *stored*
vectors via always-on pods, so a large-but-lightly-queried index becomes a top
infra cost. ChromaDB in `PersistentClient` mode stores vectors + metadata on the
local filesystem (SQLite + an HNSW index) with **no separate server and no
per-pod hourly fee** — cost is just the VM/disk you already run. It is trivial to
embed, supports metadata filtering, and is a credible drop-in for small-to-mid
corpora. See [`results/cost_comparison.md`](results/cost_comparison.md) for the
numbers, including where a managed serverless option actually wins.

---

## Setup (works in under 10 minutes)

```bash
cd problem1-rag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # then edit .env
```

### Two ways to run

**A) Fully offline (no API key, zero spend)** — great for a first smoke test. In
`.env` set:

```
EMBEDDING_PROVIDER=local
```

Local embeddings are deterministic (hashing trick), so ingestion, idempotency,
retrieval, the IR metrics, and latency all work end-to-end without a key. (Answer
generation and the LLM-as-judge still need an LLM key — see B.)

**B) Real embeddings + generation** — set your keys and provider in `.env`:

```
EMBEDDING_PROVIDER=openai
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai            # or: anthropic (needs ANTHROPIC_API_KEY)
```

> All configuration is via env vars; **no secrets are hardcoded**. `.env` is
> gitignored.

> **Switching embedding provider?** Chunk ids are content hashes (independent of
> the embedding), so re-ingesting into an existing store *skips* already-stored
> chunks and won't re-embed them with the new model. When you change
> `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`, start a fresh store — delete
> `chroma_db/` or point `CHROMA_PERSIST_DIR` at a new directory — then re-ingest.

---

## Quick start (CLI)

```bash
# 1. Ingest the bundled sample corpus (PDF/HTML/MD supported).
python -m src.cli ingest data/sample_corpus

# 2. Ask a question (top-k configurable; optional source filter).
python -m src.cli query "What is cosine similarity?" --k 3
python -m src.cli query "How does idempotent ingestion work?" --source chromadb_notes.md

# 3. Inspect the store / list chunk ids (used to build the eval gold set).
python -m src.cli stats
python -m src.cli chunks
```

## Quick start (HTTP API)

```bash
uvicorn src.api:app --port 8000
```

```bash
# Ingest
curl -s localhost:8000/ingest -H 'content-type: application/json' \
  -d '{"folder": "data/sample_corpus"}' | python -m json.tool

# Query -> {answer, cited_chunks, latency_ms, chunk_count, token_usage, ...}
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "What is cosine similarity?", "k": 3}' | python -m json.tool

# Query with a metadata filter (by source filename)
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question": "idempotent ingestion", "filter": {"source": "chromadb_notes.md"}}' \
  | python -m json.tool

# Config + collection stats
curl -s localhost:8000/health | python -m json.tool
curl -s localhost:8000/stats  | python -m json.tool
```

Every `/query` is logged (latency, chunk count, token usage) to
`logs/queries.log` as JSON lines.

---

## Requirements coverage

| Requirement | Where |
| --- | --- |
| Ingest PDF/HTML/MD | [`src/ingest.py`](src/ingest.py) (`.pdf`, `.html/.htm`, `.md/.markdown`, `.txt`) |
| Configurable chunk size + overlap (defaults 500/50) | [`src/chunker.py`](src/chunker.py) + `.env` (rationale in the docstring) |
| Idempotent re-ingest (no duplicate vectors) | deterministic `sha256(source_path + chunk_text)` id, checked before insert — [`src/ingest.py`](src/ingest.py) |
| Metadata (source, page/section, chunk_index) | stored on every chunk — [`src/ingest.py`](src/ingest.py) |
| Embed + record model + dimensionality | [`src/embed_store.py`](src/embed_store.py) (`/health`, `/stats`, eval config all report it) |
| Metadata filter at query time | `filter` in `/query`, `--source` in CLI, `where` in `retrieve()` |
| Top-k retrieval (k is a parameter) | `k` in `/query` and CLI |
| Grounded answer that cites chunks | [`src/generate.py`](src/generate.py) — inline `[source #chunk_index]` + `cited_chunks` |
| "No relevant context" without hallucinating | configurable `SIMILARITY_THRESHOLD`; below it we return `"no relevant context found"` and skip the LLM |
| HTTP endpoint + config via env + per-query logging | [`src/api.py`](src/api.py) + [`src/logging_utils.py`](src/logging_utils.py) |
| Retrieval metrics (Recall@k, MRR, nDCG, context precision) | [`eval/retrieval_metrics.py`](eval/retrieval_metrics.py) |
| Answer metrics (faithfulness/relevance + EM/F1) | [`eval/answer_metrics.py`](eval/answer_metrics.py) |
| Full eval run + results files + p50/p95 latency | [`eval/run_eval.py`](eval/run_eval.py) → `results/eval_results.json`, `results/eval_summary.md` |
| Cost comparison across scale | [`results/cost_analysis.py`](results/cost_analysis.py) → `results/cost_comparison.md` |

---

## Proving idempotency

```bash
python -m src.cli ingest data/sample_corpus   # inserts N chunks
python -m src.cli ingest data/sample_corpus   # inserts 0 (all skipped)
```

The second run reports `"chunks_inserted": 0` and `"chunks_skipped_existing": N`,
and the collection count is unchanged — because each chunk's id is a hash of
`(source_path + chunk_text)`, re-ingesting identical content is a no-op.

---

## ▶ Using YOUR corpus and running the evaluation

This is the part you'll drive yourself. Five steps:

**1. Drop your documents into a folder** (any mix of `.pdf`, `.html`, `.md`,
`.txt`), e.g. `data/my_corpus/`.

**2. Ingest them:**

```bash
python -m src.cli ingest data/my_corpus
```

**3. Find the chunk ids for your gold set.** Gold chunk ids are content hashes you
can't guess, so list every stored chunk with its human-readable
`(source, chunk_index)` and its id:

```bash
python -m src.cli chunks
```

**4. Fill in `eval/questions.json`** with 15–30 questions. For each, mark the gold
(truly relevant) chunk(s). You can reference gold in **either** way — the harness
resolves both:

```jsonc
{
  "id": "q1",
  "question": "Your question about your corpus?",
  "gold_answer": "Optional — enables Exact-Match / F1 scoring.",
  "gold_chunks":    [{ "source": "your_doc.pdf", "chunk_index": 4 }],  // human-friendly
  "gold_chunk_ids": []                                                 // or raw ids from `chunks`
}
```

The bundled examples are wired to `data/sample_corpus` so the harness runs before
you change anything — delete them and add your own.

**5. Run the evaluation:**

```bash
python -m eval.run_eval --k 5
```

This writes:
- `results/eval_results.json` — config, **per-question** metrics, aggregates, warnings.
- `results/eval_summary.md` — human-readable tables (retrieval, answer, latency).

It prints Recall@k / Hit Rate / MRR / nDCG / Context Precision, LLM-judge
faithfulness + relevance (and EM/F1 if you gave gold answers), and **p50/p95
retrieval latency**.

> **Offline note:** with no LLM key, `run_eval` still computes retrieval metrics
> and latency and clearly marks the answer-quality section as skipped. With a key
> set, it also generates answers and runs the LLM-as-judge.

---

## Project layout

```
problem1-rag/
├── src/
│   ├── config.py         # env-driven config (no hardcoded secrets)
│   ├── chunker.py        # token-based chunking (tiktoken), size + overlap
│   ├── ingest.py         # load PDF/HTML/MD, chunk, deterministic ids, idempotent
│   ├── embed_store.py    # OpenAI + local embeddings; ChromaDB (cosine) wrapper
│   ├── retrieve.py       # top-k retrieval + threshold policy + metadata filter
│   ├── generate.py       # grounded, cited generation; no-context guard
│   ├── api.py            # FastAPI: /ingest, /query, /health, /stats
│   ├── cli.py            # ingest / query / stats / chunks
│   └── logging_utils.py  # per-query JSONL logging
├── eval/
│   ├── questions.json    # your 15-30 Q&A + gold chunks (template + examples)
│   ├── retrieval_metrics.py  # Recall@k, Hit Rate, MRR, nDCG, context precision
│   ├── answer_metrics.py     # LLM-as-judge + EM/F1
│   └── run_eval.py           # runs everything -> results/*
├── results/
│   ├── cost_analysis.py      # generates the cost table
│   ├── cost_comparison.md    # generated
│   ├── eval_results.json     # generated
│   └── eval_summary.md       # generated
├── data/sample_corpus/   # runnable-out-of-the-box sample docs
├── requirements.txt
├── .env.example
└── README.md
```

---

## Discussion

### When would you switch back to a managed vector DB?

The cost model in [`results/cost_comparison.md`](results/cost_comparison.md) makes
the boundary concrete. Self-hosted ChromaDB is the right call for a **large,
lightly-queried index** — it crushes always-on managed *pods*, whose cost scales
with stored vectors regardless of traffic (e.g. ~$1,600/mo for 10M vectors on pods
vs a single ~$800/mo VM). I'd switch back to a managed DB when:

- **Query volume is high.** Once a VM is paid for, its reads are effectively free;
  a managed *serverless* DB charges per query. The sensitivity table shows the
  crossover — for a 1M-vector index, serverless is far cheaper when lightly queried
  but is overtaken by the flat VM somewhere around tens of millions of
  queries/month.
- **The index outgrows one machine's RAM**, or I need horizontal scale-out without
  building sharding myself.
- **I need managed HA/replication, autoscaling, backups, and multi-region** rather
  than owning VM patching, monitoring, and on-call. That operational load is what
  the managed premium is partly buying, and past a certain team/SLA point it's
  cheaper than an engineer's time.
- **Bursty or unpredictable traffic** where a serverless pay-per-use model beats
  paying for a VM sized for peak.

In short: ChromaDB for cost-sensitive, steady, single-node-sized workloads; managed
when scale, availability, or operational simplicity dominate the bill.

### Was retrieval or generation the weaker link?

Read it off your own `results/eval_summary.md`:

- **Retrieval is the weak link if** Recall@k / Hit Rate / nDCG are low — the right
  chunks aren't reaching the model, so even a perfect generator can't answer. Fixes
  live upstream: chunk size/overlap, a better embedding model, higher `k`, hybrid
  (keyword + vector) search, or reranking.
- **Generation is the weak link if** retrieval metrics are strong (gold chunks are
  in the top-k) but the LLM-judge **faithfulness** is low — the model has the
  evidence but drifts from it or fabricates. Fixes are downstream: a stricter
  grounding prompt, a smaller/tighter context window, lower temperature, or a
  stronger model.

On the bundled sample corpus, retrieval is essentially saturated (Recall@k = 1.0,
nDCG ≈ 0.93 — the gold chunk is always retrieved because the corpus is tiny), so
any quality gap would sit in generation/faithfulness. On a real corpus the
diagnostic is the same: if Hit Rate is high but faithfulness is low, invest in the
prompt/model; if Hit Rate itself is low, invest in retrieval. Context Precision is
low here (0.33) only because each question has one gold chunk out of `k=3`
retrieved — expected, not a defect.

---

## Trade-offs accepted

- **Single-node store.** No built-in HA/replication; the index is bounded by one
  machine's RAM. We accept this for the cost win at small/mid scale.
- **We self-manage the VM** (patching, backups, monitoring) — real operational cost
  the managed price would otherwise cover.
- **Local embedding fallback is for testing, not production quality** — it exists so
  the pipeline is runnable and CI-testable without spend; use `openai` for real
  retrieval quality.
