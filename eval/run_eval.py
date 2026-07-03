"""End-to-end evaluation harness.

Runs the fixed question set and reports all three evaluation layers:

  * Retrieval:  Recall@k, Hit Rate@k, MRR@k, nDCG@k, Context Precision@k
  * Answer:     LLM-as-judge faithfulness + relevance (1-5), plus EM/F1 when a
                gold_answer is provided.
  * Latency:    p50 / p95 retrieval latency across the question set.

Outputs:
  results/eval_results.json  (machine-readable: config, per-question, aggregates)
  results/eval_summary.md    (human-readable summary table)

Gold chunks are specified in questions.json by human-friendly (source, chunk_index)
pairs and/or raw chunk ids; this harness resolves them against the live store.

Usage:
  python -m eval.run_eval                 # uses env config (TOP_K, threshold, ...)
  python -m eval.run_eval --k 5 --questions eval/questions.json

Runs offline too: with no LLM key configured, retrieval + latency are still
computed and the answer-quality section is clearly marked as skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# Make `src` and `eval` importable regardless of CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from eval import answer_metrics as am  # noqa: E402
from eval import retrieval_metrics as rm  # noqa: E402
from src.config import settings  # noqa: E402
from src.embed_store import VectorStore, embedding_info  # noqa: E402
from src.generate import generate_answer  # noqa: E402
from src.retrieve import retrieve  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "results")


def _llm_available() -> bool:
    if settings.llm_provider == "openai":
        return bool(settings.openai_api_key)
    if settings.llm_provider == "anthropic":
        return bool(settings.anthropic_api_key)
    return False


def _build_gold_index(store: VectorStore) -> dict[tuple[str, int], str]:
    """Map (source, chunk_index) -> chunk_id for resolving human-friendly gold."""
    index: dict[tuple[str, int], str] = {}
    for row in store.all_chunks():
        key = (row["source"], row["chunk_index"])
        index[key] = row["chunk_id"]
    return index


def _resolve_gold(question: dict, gold_index: dict[tuple[str, int], str]) -> tuple[set[str], list[str]]:
    """Resolve a question's gold references to a set of chunk ids.

    Returns (gold_ids, warnings)."""
    gold: set[str] = set()
    warnings: list[str] = []

    for cid in question.get("gold_chunk_ids", []) or []:
        gold.add(cid)

    for ref in question.get("gold_chunks", []) or []:
        key = (ref.get("source"), ref.get("chunk_index"))
        cid = gold_index.get(key)
        if cid is None:
            warnings.append(
                f"gold ref {key} did not resolve to a stored chunk "
                f"(is the corpus ingested and is chunk_index correct?)"
            )
        else:
            gold.add(cid)
    return gold, warnings


def run(questions_path: str, k: int, threshold: float) -> dict:
    store = VectorStore()
    if store.count() == 0:
        raise RuntimeError(
            "The vector store is empty. Ingest a corpus first, e.g.\n"
            "  python -m src.cli ingest data/sample_corpus"
        )

    with open(questions_path, encoding="utf-8") as f:
        spec = json.load(f)
    questions = [q for q in spec.get("questions", []) if "question" in q]
    if not questions:
        raise RuntimeError(f"No questions found in {questions_path}")

    gold_index = _build_gold_index(store)
    llm_on = _llm_available()

    per_question_results: list[dict] = []
    retrieval_metric_rows: list[dict] = []
    retrieval_latencies_ms: list[float] = []
    em_scores: list[float] = []
    f1_scores: list[float] = []
    faithfulness_scores: list[int] = []
    relevance_scores: list[int] = []
    all_warnings: list[str] = []

    for q in questions:
        qid = q.get("id", q["question"][:40])
        gold, warnings = _resolve_gold(q, gold_index)
        for w in warnings:
            all_warnings.append(f"[{qid}] {w}")

        # --- Retrieval (timed for the latency stats) ---
        t0 = time.perf_counter()
        hits = retrieve(q["question"], k=k, store=store)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        retrieval_latencies_ms.append(latency_ms)
        retrieved_ids = [h.id for h in hits]

        rmetrics = rm.per_question_metrics(retrieved_ids, gold, k)
        retrieval_metric_rows.append(rmetrics)

        entry: dict = {
            "id": qid,
            "question": q["question"],
            "gold_chunk_ids": sorted(gold),
            "retrieved_chunk_ids": retrieved_ids,
            "retrieval_latency_ms": round(latency_ms, 2),
            "retrieval_metrics": rmetrics,
        }

        # --- Answer generation + answer metrics (only if an LLM is configured) ---
        if llm_on:
            result = generate_answer(q["question"], k=k, threshold=threshold, store=store)
            entry["answer"] = result.answer
            entry["cited_chunk_ids"] = [c["chunk_id"] for c in result.cited_chunks]
            entry["token_usage"] = result.token_usage
            entry["no_relevant_context"] = result.no_relevant_context

            # LLM-as-judge (skip when there was no context to ground on).
            if result.cited_chunks:
                context = "\n\n".join(
                    f"[{c['source']} #{c['chunk_index']}]\n{c['text']}"
                    for c in result.cited_chunks
                )
                judge = am.llm_judge(q["question"], context, result.answer)
                entry["judge"] = judge
                if isinstance(judge.get("faithfulness"), int):
                    faithfulness_scores.append(judge["faithfulness"])
                if isinstance(judge.get("answer_relevance"), int):
                    relevance_scores.append(judge["answer_relevance"])
            else:
                entry["judge"] = {"note": "no context retrieved above threshold; judge skipped"}

            # EM / F1 vs gold answer, if provided.
            gold_answer = q.get("gold_answer")
            if gold_answer:
                em = am.exact_match(result.answer, gold_answer)
                f1 = am.token_f1(result.answer, gold_answer)
                entry["em"] = em
                entry["f1"] = round(f1, 4)
                em_scores.append(em)
                f1_scores.append(f1)

        per_question_results.append(entry)

    # --- Aggregates ---
    agg_retrieval = rm.aggregate_metrics(retrieval_metric_rows)
    lat = np.array(retrieval_latencies_ms)
    latency_summary = {
        "count": int(lat.size),
        "p50_ms": round(float(np.percentile(lat, 50)), 2),
        "p95_ms": round(float(np.percentile(lat, 95)), 2),
        "mean_ms": round(float(lat.mean()), 2),
        "max_ms": round(float(lat.max()), 2),
    }

    answer_summary = {"llm_judge_enabled": llm_on}
    if llm_on:
        answer_summary["mean_faithfulness_1to5"] = (
            round(sum(faithfulness_scores) / len(faithfulness_scores), 3)
            if faithfulness_scores else None
        )
        answer_summary["mean_answer_relevance_1to5"] = (
            round(sum(relevance_scores) / len(relevance_scores), 3)
            if relevance_scores else None
        )
        answer_summary["judged_questions"] = len(faithfulness_scores)
        if em_scores:
            answer_summary["exact_match"] = round(sum(em_scores) / len(em_scores), 3)
            answer_summary["token_f1"] = round(sum(f1_scores) / len(f1_scores), 3)
            answer_summary["gold_answer_questions"] = len(em_scores)

    return {
        "config": {
            "k": k,
            "similarity_threshold": threshold,
            "embedding": embedding_info(),
            "llm_provider": settings.llm_provider,
            "llm_model": (
                settings.openai_llm_model
                if settings.llm_provider == "openai"
                else settings.anthropic_llm_model
            ),
            "num_questions": len(questions),
            "num_chunks_in_store": store.count(),
        },
        "retrieval_metrics": {kk: round(vv, 4) for kk, vv in agg_retrieval.items()},
        "answer_metrics": answer_summary,
        "latency": latency_summary,
        "warnings": all_warnings,
        "per_question": per_question_results,
    }


def _write_summary_md(report: dict, path: str) -> None:
    cfg = report["config"]
    rmx = report["retrieval_metrics"]
    ans = report["answer_metrics"]
    lat = report["latency"]

    lines: list[str] = []
    lines.append("# RAG Evaluation Summary\n")
    lines.append("## Configuration\n")
    lines.append(f"- Questions: **{cfg['num_questions']}**  |  Chunks in store: **{cfg['num_chunks_in_store']}**")
    lines.append(f"- Retrieval k: **{cfg['k']}**  |  Similarity threshold: **{cfg['similarity_threshold']}**")
    lines.append(f"- Embedding: **{cfg['embedding']['model']}** ({cfg['embedding']['dimensionality']}-dim, provider={cfg['embedding']['provider']})")
    lines.append(f"- LLM: **{cfg['llm_provider']} / {cfg['llm_model']}**\n")

    lines.append("## Retrieval quality (mean across questions)\n")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")
    label = {
        "recall@k": f"Recall@{cfg['k']}",
        "hit_rate@k": f"Hit Rate@{cfg['k']}",
        "mrr@k": f"MRR@{cfg['k']}",
        "ndcg@k": f"nDCG@{cfg['k']}",
        "context_precision@k": f"Context Precision@{cfg['k']}",
    }
    for key, val in rmx.items():
        lines.append(f"| {label.get(key, key)} | {val:.4f} |")
    lines.append("")

    lines.append("## Answer quality\n")
    if not ans.get("llm_judge_enabled"):
        lines.append("_LLM-as-judge disabled (no LLM API key configured). "
                     "Retrieval + latency were still evaluated._\n")
    else:
        lines.append("| Metric | Value |")
        lines.append("| --- | --- |")
        lines.append(f"| Faithfulness (1-5) | {ans.get('mean_faithfulness_1to5')} |")
        lines.append(f"| Answer relevance (1-5) | {ans.get('mean_answer_relevance_1to5')} |")
        if "exact_match" in ans:
            lines.append(f"| Exact Match | {ans['exact_match']} |")
            lines.append(f"| Token F1 | {ans['token_f1']} |")
        lines.append(f"\n_Judged {ans.get('judged_questions', 0)} question(s)._\n")

    lines.append("## Retrieval latency\n")
    lines.append("| Metric | ms |")
    lines.append("| --- | --- |")
    lines.append(f"| p50 | {lat['p50_ms']} |")
    lines.append(f"| p95 | {lat['p95_ms']} |")
    lines.append(f"| mean | {lat['mean_ms']} |")
    lines.append(f"| max | {lat['max_ms']} |")
    lines.append("")

    if report["warnings"]:
        lines.append("## Warnings\n")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG evaluation harness.")
    parser.add_argument("--questions", default=os.path.join(_ROOT, "eval", "questions.json"))
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--threshold", type=float, default=settings.similarity_threshold)
    args = parser.parse_args()

    report = run(args.questions, k=args.k, threshold=args.threshold)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, "eval_results.json")
    md_path = os.path.join(RESULTS_DIR, "eval_summary.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    _write_summary_md(report, md_path)

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}\n")
    print("Retrieval:", json.dumps(report["retrieval_metrics"]))
    print("Answer:   ", json.dumps(report["answer_metrics"]))
    print("Latency:  ", json.dumps(report["latency"]))
    if report["warnings"]:
        print(f"\n{len(report['warnings'])} warning(s) — see results/eval_summary.md")


if __name__ == "__main__":
    main()
