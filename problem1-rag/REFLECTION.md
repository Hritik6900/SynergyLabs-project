# Reflection

> _Written in my own words — a short retrospective on building this RAG service:
> the decisions I made, what broke, and what I'd change._

## What I set out to build

The brief's core claim is that managed vector DBs bill you for *stored* vectors via
always-on pods, so a large but lightly-queried index becomes a top infra cost. I
wanted to prove a low-cost store is a credible alternative **with numbers**, not
just assert it. So I picked **ChromaDB** (embedded, on-disk, no server process) and
built the whole QA pipeline around honest measurement: retrieval quality, answer
quality, latency, and cost.

## Key design decisions (and the tradeoffs I accepted)

- **ChromaDB persistent client** over a managed DB — the cost win is real (a fixed
  VM vs per-pod hourly fees), and I accepted the tradeoffs: single-node, no built-in
  HA/replication, and I manage the box myself.
- **Local `sentence-transformers` embeddings + Groq for generation** — this makes
  the pipeline *free to embed* (no OpenAI dependency) and cheap to run. Embeddings
  are the part you call most, so keeping them local matters.
- **Content-addressed chunk ids** (`sha256(source_path + chunk_text)`) — this is what
  makes re-ingest idempotent without a separate dedup table. Simple and it just
  works.
- **A configurable similarity threshold** so the system returns "no relevant context
  found" instead of hallucinating when nothing is close enough.

## What broke (and how I dealt with it)

The most instructive failure was the **Groq free-tier rate limit**. My evaluation
harness ran fine on a handful of questions, then on the full set it **crashed
mid-run** with an unhelpful traceback. When I captured stderr properly, the real
cause was a `429 … tokens per day (TPD): Limit 100000, Used 98870` — I'd exhausted
the daily token budget on `llama-3.3-70b`, partly because my document chunks are
full of ASCII diagrams and tables that are surprisingly token-heavy.

That one bug taught me three things and drove three fixes:

1. **A single failed API call shouldn't nuke a whole eval run.** I made the harness
   *resilient*: retrieval metrics (which need no LLM) are always computed for every
   question, and a per-question LLM failure is recorded and skipped, not fatal.
2. **Distinguish per-minute from per-day limits.** I added retry-with-backoff that
   retries transient/per-minute rate limits but *fails fast* on a daily cap (no point
   sleeping for hours), plus a request timeout so a call can't hang forever.
3. **Give myself a knob for constrained tiers.** I added `--answer-sample N` so the
   expensive LLM-judge can run on a representative subset while retrieval still
   covers all questions, and a context-truncation option to bound prompt tokens.

The lesson: on free/rate-limited infra, *graceful degradation* matters as much as the
happy path. The system now produces partial, honest results under a quota instead of
dying.

## A surprise: I couldn't trust my own gold labels

I hand-labelled which chunk is "relevant" for each eval question by eyeballing the
docs. When I actually verified them against the chunker, **22 of my ~28 estimates
were wrong.** The reason is subtle: my docs have code blocks, tables, and box-drawing
characters that tokenize very differently from words, so "the 6th section" is not the
"6th chunk." I wrote a small `verify_gold.py` that locates each answer's exact phrase
in the real chunks and corrects the index. Without that step my retrieval metrics
would have been silently, completely wrong — a good reminder that **evaluation code
needs the same rigor as the system it measures.**

## An experiment that failed (and why that's useful)

Once I saw retrieval was the weak link (Recall@5 ≈ 0.73), the obvious move was "use a
bigger embedding model." I tested `all-mpnet-base-v2` (768-dim, 2× larger than
MiniLM) — and it made things **slightly worse** (Recall 0.73→0.70). Digging into the
misses showed why: they're overwhelmingly *exact-term* questions (config keys,
endpoints, numeric thresholds) sitting inside tables — and no *dense* model, however
large, has a keyword signal to match those. The real fix is hybrid (BM25 + vector)
search or a retrieval-tuned model, not a bigger encoder. I kept MiniLM (it won) and
documented the null result, because a tested "this doesn't work, here's why" is worth
more than an untested guess.

## What I'd do with more time

- **Hybrid retrieval (BM25 + dense)** — the single change most likely to lift Recall
  on these identifier-heavy technical docs.
- **Header-aware / structure-aware chunking** so ASCII diagrams and tables don't
  dilute a chunk's embedding.
- A small **reranker** (cross-encoder) over the top-k to fix the "intro chunk
  outranks the specific chunk" problem I saw in MRR.
- Benchmark a third store (I already benchmarked FAISS vs ChromaDB) and run the full
  answer-eval on the 70B model once quota resets.

## What I learned

The building blocks of a RAG system are easy; the honesty is the hard part. Most of
my time went into *trusting the numbers* — verifying gold labels, making the harness
survive real-world API limits, and testing my assumptions (the embedding swap) rather
than asserting them. That's the part I'm most confident about in this submission.
