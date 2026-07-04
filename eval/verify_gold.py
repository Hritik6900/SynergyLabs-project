"""Verify (and optionally fix) gold chunk_index values in questions.json.

Your corpus has ASCII diagrams / tables / code blocks that tokenize unevenly, so a
human-estimated chunk_index can be wrong. Each question may carry a `_locate_hint`
(an exact phrase from its gold chunk). This tool ingests nothing itself — it reads
the live store — locates the chunk(s) that actually contain each hint, and reports
whether the recorded chunk_index matches. With --fix it rewrites the indices.

Usage:
  python -m eval.verify_gold             # report only
  python -m eval.verify_gold --fix       # correct chunk_index in questions.json

Matching: whitespace-normalized, case-insensitive substring match. A hint may hold
several fragments separated by '...'; all must appear in the same chunk. If the
phrase spans an overlap boundary it can match two adjacent chunks — both are valid
gold; we keep the recorded one if it matches, else the lowest-index match.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.embed_store import VectorStore  # noqa: E402

QUESTIONS = os.path.join(_ROOT, "eval", "questions.json")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def _load_full_chunks() -> dict[str, list[tuple[int, str]]]:
    """source -> [(chunk_index, normalized_full_text), ...]."""
    store = VectorStore()
    got = store._collection.get(include=["documents", "metadatas"])
    by_source: dict[str, list[tuple[int, str]]] = {}
    for doc, meta in zip(got["documents"], got["metadatas"]):
        by_source.setdefault(meta["source"], []).append((meta["chunk_index"], _norm(doc)))
    for src in by_source:
        by_source[src].sort(key=lambda t: t[0])
    return by_source


def _find(hint: str, chunks: list[tuple[int, str]]) -> tuple[list[int], list[int]]:
    """Return (full_matches, partial_matches).

    full_matches: chunks containing every '...'-separated fragment.
    partial_matches: chunks containing the single longest fragment (fallback).
    """
    frags = [_norm(f) for f in hint.split("...") if f.strip()]
    full = [idx for idx, text in chunks if all(f in text for f in frags)]
    longest = max(frags, key=len) if frags else ""
    partial = [idx for idx, text in chunks if longest and longest in text]
    return full, partial


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify/fix gold chunk indices.")
    ap.add_argument("--fix", action="store_true", help="rewrite corrected indices into questions.json")
    args = ap.parse_args()

    by_source = _load_full_chunks()
    if not by_source:
        raise SystemExit("Store is empty — ingest the corpus first (python -m src.cli ingest ...).")

    spec = json.load(open(QUESTIONS, encoding="utf-8"))
    questions = spec["questions"]

    ok = fixed = unhinted = notfound = 0
    print(f"{'id':<32} {'source':<22} {'rec':>4} {'found':>10}  status")
    print("-" * 84)
    for q in questions:
        gc = q["gold_chunks"][0]
        src, rec = gc["source"], gc["chunk_index"]
        hint = q.get("_locate_hint")
        chunks = by_source.get(src, [])

        if not hint:
            unhinted += 1
            print(f"{q['id']:<32} {src:<22} {rec:>4} {'—':>10}  no-hint (trusted)")
            continue

        full, partial = _find(hint, chunks)
        if full:
            chosen = rec if rec in full else full[0]
            if chosen == rec:
                ok += 1
                print(f"{q['id']:<32} {src:<22} {rec:>4} {str(full):>10}  OK")
            else:
                fixed += 1
                print(f"{q['id']:<32} {src:<22} {rec:>4} {str(full):>10}  FIX -> {chosen}")
                if args.fix:
                    gc["chunk_index"] = chosen
        elif partial:
            chosen = rec if rec in partial else partial[0]
            status = "OK(partial)" if chosen == rec else f"FIX(partial) -> {chosen}"
            if chosen != rec:
                fixed += 1
                if args.fix:
                    gc["chunk_index"] = chosen
            else:
                ok += 1
            print(f"{q['id']:<32} {src:<22} {rec:>4} {str(partial):>10}  {status}")
        else:
            notfound += 1
            print(f"{q['id']:<32} {src:<22} {rec:>4} {'[]':>10}  NOT FOUND (check hint)")

    print("-" * 84)
    print(f"OK={ok}  needs-fix={fixed}  no-hint={unhinted}  not-found={notfound}  total={len(questions)}")

    if args.fix and fixed:
        with open(QUESTIONS, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"\nWrote corrected indices to {QUESTIONS}")
    elif fixed and not args.fix:
        print("\nRe-run with --fix to apply the corrections above.")


if __name__ == "__main__":
    main()
