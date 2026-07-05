# Applied AI / ML Engineering — Take-Home Assignment

Two independent projects in one repository. Each has its own README, setup, and results.

| Problem | What it is | Folder |
|---|---|---|
| **1 — Cost-Efficient RAG** | A QA service over a document corpus on a low-cost vector store (ChromaDB), with honest evaluation of retrieval quality, answer quality, latency, and cost. | [`problem1-rag/`](./problem1-rag/) |
| **2 — LLM-as-Judge** | A judging pipeline that turns `{input, output, criteria}` into a structured quality verdict, and takes the judge's own biases (position, verbosity, self-enhancement, sycophancy, clustering) seriously. | [`problem2-judge/`](./problem2-judge/) |

Each project is self-contained with its own virtual environment and dependencies. Pick a folder and follow its README.

---

## Quick start

**Problem 1 — RAG**
```bash
cd problem1-rag
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY
python -m src.cli ingest data/sample_corpus
python -m src.cli query "What AI model does TerraVision use and for what?" --k 3
```

**Problem 2 — LLM-as-Judge**
```bash
cd problem2-judge
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # add GROQ_API_KEY
python main.py
```

---

## At a glance

**Problem 1 (RAG):** ChromaDB (embedded, no always-on pods) · local `all-MiniLM-L6-v2` embeddings (free) · Groq for generation · idempotent content-hash re-ingest · eval over 30 questions — Recall@5 **0.73**, MRR **0.49**, nDCG@5 **0.55**; LLM-judge faithfulness/relevance **5.0/5.0**; retrieval p50 **~21 ms**. Bonus: FAISS second-store benchmark + 26 pytest tests.

**Problem 2 (Judge):** pointwise rubric (5 weighted criteria) + pairwise A/B · judge (`gpt-oss`) and generator (`qwen`) from **different model families** · five bias mitigations with measured before→after · judge validation: gold agreement **100%** (κ=1.0), test-retest flip rate **0%** (3 runs), not fooled by verbose-but-wrong / terse-but-correct probes.

All numbers are reproducible from each project's `results/` directory.
