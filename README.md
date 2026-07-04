# Problem 1 — Cost-Efficient RAG Application

A question-answering service over a document corpus, backed by a **low-cost local
vector store (ChromaDB)**, with honest evaluation of retrieval quality, answer
quality, latency, and cost.

- **Store:** ChromaDB persistent client (embedded, on-disk, no server process).
- **Embeddings** (`EMBEDDING_PROVIDER`):
  - `sentence-transformers` — local CPU model (`all-MiniLM-L6-v2`, 384-dim), free, no key **(default)**.
  - `openai` — `text-embedding-3-small` (1536-dim).
  - `local` — deterministic hash embedding, offline/zero-deps, low quality (smoke tests only).
- **Generation** (`LLM_PROVIDER`): `groq` (`llama-3.3-70b-versatile`, default) ·
  `openai` (`gpt-4o-mini`) · `anthropic` (`claude-haiku-4-5`). Groq is chat-only
  (OpenAI-compatible); it has **no embeddings endpoint**, which is why embeddings
  are handled separately.
- **Default stack is fully free to embed:** local sentence-transformers embeddings
  + Groq generation → the only paid call is Groq generation (and Groq has a free tier).
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
# CPU-only torch first keeps the sentence-transformers install small:
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env          # then edit .env
```

### Recommended: Groq generation + local sentence-transformers embeddings

The default `.env.example` is already set to this. Just add a Groq key
(free tier at <https://console.groq.com>):

```
EMBEDDING_PROVIDER=sentence-transformers   # free, local, no key
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
```

Embeddings run locally on CPU (zero API cost); only generation calls Groq. The
first query loads the embedding model once (~a few seconds), then per-query
retrieval is ~20-35 ms.

### Other supported combinations

- **Fully offline smoke test (no key at all):** `EMBEDDING_PROVIDER=local`. Ingestion,
  idempotency, retrieval, IR metrics, and latency all run; generation + LLM-judge
  are skipped (they need an LLM key). Hash embeddings are low quality — smoke tests only.
- **OpenAI embeddings and/or generation:** `EMBEDDING_PROVIDER=openai` + `OPENAI_API_KEY`,
  and/or `LLM_PROVIDER=openai`. `text-embedding-3-small` is best-quality and costs ~cents.
- **Anthropic generation:** `LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY`.

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

# On a rate-limited free LLM tier, evaluate answers on an evenly-spread sample
# (retrieval metrics still cover ALL questions):
python -m eval.run_eval --k 5 --answer-sample 10
```

> First verify your gold chunk ids resolve to the right chunks (essential — the
> heavy ASCII/table/code content in these docs tokenizes unevenly, so estimated
> indices are often wrong): `python -m eval.verify_gold` (add `--fix` to correct).

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

## Tests

A hermetic pytest suite (no API key / network — uses the deterministic local
embedder and throwaway temp stores) covers the tricky logic:

```bash
pip install pytest
python -m pytest          # 26 tests: chunking, idempotency, IR metrics, EM/F1,
                          # store add/query/filter, no-context guard, FAISS parity
```

Covered: chunk sizing/overlap/coverage, deterministic ids + idempotent re-ingest,
Recall@k / MRR / nDCG / context-precision against hand-computed values, EM/F1
normalization, metadata-filtered retrieval, the "no relevant context" guard, and
ChromaDB-vs-FAISS top-k agreement.

## Bonus: benchmarking a second store (FAISS)

Beyond the required single store, a second backend ([`src/faiss_store.py`](src/faiss_store.py),
an in-memory **exact** `IndexFlatIP`) is benchmarked against ChromaDB's approximate
HNSW on the **same embeddings + same query vectors**:

```bash
python results/store_benchmark.py --k 5 --repeats 50   # -> results/store_benchmark.md
```

It reports per-search **p50/p95 latency** for each store and **recall agreement@k**
(how much of FAISS's exact top-k ChromaDB's approximate index also returns — i.e.
what the approximation costs). See [`results/store_benchmark.md`](results/store_benchmark.md).
Takeaway: FAISS is faster raw but is *just an index* (no persistence, metadata, or
filtering); ChromaDB bundles those, which is why it's the primary store and FAISS
is the yardstick.

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
│   ├── llm_client.py     # one entry point for openai / groq / anthropic chat
│   ├── faiss_store.py    # second store backend (exact) for the benchmark
│   ├── api.py            # FastAPI: /ingest, /query, /health, /stats
│   ├── cli.py            # ingest / query / stats / chunks
│   └── logging_utils.py  # per-query JSONL logging
├── tests/                # hermetic pytest suite (26 tests, no key needed)
├── eval/
│   ├── questions.json    # 30 verified Q&A + gold chunks (source, chunk_index)
│   ├── verify_gold.py    # locates each gold chunk by phrase; --fix corrects indices
│   ├── retrieval_metrics.py  # Recall@k, Hit Rate, MRR, nDCG, context precision
│   ├── answer_metrics.py     # LLM-as-judge + EM/F1
│   └── run_eval.py           # runs everything -> results/*
├── results/
│   ├── cost_analysis.py      # generates the cost table
│   ├── cost_comparison.md    # generated
│   ├── store_benchmark.py    # ChromaDB vs FAISS benchmark (bonus)
│   ├── store_benchmark.md    # generated
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

**On this corpus (4 project READMEs, 63 chunks, 30 verified questions, k=5),
retrieval is the weak link — decisively.** The committed run in
[`results/eval_summary.md`](results/eval_summary.md) shows:

| Layer | Metric | Value |
| --- | --- | --- |
| Retrieval (all 30 Q) | Recall@5 / Hit Rate | **0.73** |
| | MRR@5 | **0.49** |
| | nDCG@5 | 0.55 |
| Answer (LLM-judge, 8-Q sample) | Faithfulness (1-5) | **5.0** |
| | Answer relevance (1-5) | **5.0** |

For ~8 of 30 questions the gold chunk never entered the top-5 (Recall 0.73), and
MRR 0.49 means even when it *is* retrieved it's often not rank 1. But when the right
context is retrieved, generation is strong (faithfulness 5.0/5, relevance 5.0/5).
**So the bottleneck is retrieval, not generation** — the fix is upstream: a stronger
embedding model (e.g. `all-mpnet-base-v2`, 768-dim), higher `k`, smaller/cleaner
chunks (these README chunks are dense with ASCII diagrams and tables that embed
poorly), or hybrid keyword+vector search with reranking.

Notes on the numbers: (1) retrieval metrics cover **all 35** questions; the
LLM-judge answer-eval ran on an **evenly-spread 8-question sample** to fit the Groq
free-tier rate limit (`--answer-sample`; see below) and used `llama-3.1-8b-instant`
— re-run with the 70B model for higher answer scores once daily quota resets.
(2) EM = 0.0 with F1 = 0.36 is expected: the model paraphrases correct answers
rather than matching the terse gold strings, so faithfulness (semantic) is the more
informative answer metric. (3) Context Precision = 0.15 is expected — each question
has a single gold chunk out of `k=5` retrieved.

---

## Trade-offs accepted

- **Single-node store.** No built-in HA/replication; the index is bounded by one
  machine's RAM. We accept this for the cost win at small/mid scale.
- **We self-manage the VM** (patching, backups, monitoring) — real operational cost
  the managed price would otherwise cover.
- **Local embedding fallback is for testing, not production quality** — it exists so
  the pipeline is runnable and CI-testable without spend; use `openai` for real
  retrieval quality.
