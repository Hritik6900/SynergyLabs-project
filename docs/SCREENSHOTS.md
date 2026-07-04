# Screenshots

Two screenshots demonstrate the system end-to-end. Run each command yourself and
paste a terminal screenshot in the placeholder. The **real captured output** is
shown under each command so you know exactly what to expect (and can verify your
run matches).

> Setup (once): `cd problem1-rag && source .venv/bin/activate` and make sure
> `.env` has a `GROQ_API_KEY`. Ingest the corpus first if you haven't:
> `python -m src.cli ingest data/sample_corpus`.

---

## 1. End-to-end query (retrieval → grounded, cited answer)

```bash
python -m src.cli query "What AI model does TerraVision use and for what?" --k 3
```

**Expected output** (captured live):

```
=== ANSWER ===
TerraVision uses the Google Gemini AI model, specifically the `gemini-2.5-flash`
variant, for text analysis and image generation. It uses text analysis for green
corridor planning and urban metrics analysis, and image generation for transforming
street photos with sustainability interventions. [terra_vision.md #1]

=== CITED CHUNKS ===
  [terra_vision.md #0] sim=0.6417 id=a2a6642d0fdd...
  [NavDrishti-Server.md #0] sim=0.4616 id=eb3aef8fac70...
  [DualCast.md #0] sim=0.4007 id=fae4146fd81a...

=== METRICS ===
{
  "chunk_count": 3,
  "token_usage": { "prompt_tokens": 1658, "completion_tokens": 63, "total_tokens": 1721 },
  "retrieval_latency_ms": 11236.36,
  "generation_latency_ms": 1002.14,
  "no_relevant_context": false
}
```

What this shows: top-k retrieval, a **grounded answer with an inline citation**
(`[terra_vision.md #1]`), the **cited chunks** with similarity scores, and
**token usage + latency** logged per query. (The ~11s `retrieval_latency_ms` here is
the one-time embedding-model load on the *first* query of a process; subsequent
queries are ~20–30 ms.)

> _Paste your terminal screenshot here:_
>
> ![End-to-end query](./img/query.png)

You can also show the **HTTP API** version (start `uvicorn src.api:app --port 8000`):

```bash
curl -s localhost:8000/query -H 'content-type: application/json' \
  -d '{"question":"What AI model does TerraVision use?","k":3}' | python -m json.tool
```

> _Optional API screenshot:_
>
> ![API query](./img/api_query.png)

---

## 2. Idempotent re-ingest (no duplicate vectors)

```bash
python -m src.cli ingest data/sample_corpus   # first ingest
python -m src.cli ingest data/sample_corpus   # second ingest (idempotent)
```

**Expected output** (captured live):

```
$ python -m src.cli ingest data/sample_corpus   # first ingest
  "chunks_total": 63,
  "chunks_inserted": 63,
  "chunks_skipped_existing": 0,
Collection now holds 63 chunks.

$ python -m src.cli ingest data/sample_corpus   # second ingest (idempotent)
  "chunks_total": 63,
  "chunks_inserted": 0,
  "chunks_skipped_existing": 63,
Collection now holds 63 chunks.
```

What this shows: the first run inserts 63 chunks; the second run inserts **0** and
**skips all 63** — the collection count is unchanged. Because each chunk id is a
`sha256(source_path + chunk_text)` hash, re-ingesting identical content is a no-op,
so there are **no duplicate vectors**.

> _Paste your terminal screenshot here:_
>
> ![Idempotent re-ingest](./img/idempotent.png)

---

### How to add the images

Save your screenshots into `docs/img/` with the filenames referenced above
(`query.png`, `idempotent.png`, optionally `api_query.png`). The `![...](...)`
tags will then render them inline on GitHub.
