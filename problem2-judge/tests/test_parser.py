"""Offline tests for the robust JSON parser (no API key / network needed).

Run: python -m tests.test_parser   (from project root)
Exercises the non-LLM strategies (1-4) on intentionally-messy responses that
open-weight models on Groq realistically emit.
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parser import parse_json_best_effort

MESSY_SAMPLES = [
    # (name, raw_text, expected_key_present)
    ("clean", '{"verdict": "pass", "overall_score": 4.2}', "verdict"),
    ("markdown_fence", '```json\n{"verdict": "fail", "overall_score": 1.0}\n```', "verdict"),
    ("fence_no_lang", '```\n{"verdict": "pass"}\n```', "verdict"),
    ("leading_prose", 'Sure! Here is my evaluation:\n{"verdict": "pass", "overall_score": 5}', "verdict"),
    ("trailing_text", '{"verdict": "borderline"}\nHope this helps!', "verdict"),
    ("trailing_comma", '{"verdict": "pass", "overall_score": 4,}', "verdict"),
    ("single_quotes", "{'verdict': 'fail', 'overall_score': 2}", "verdict"),
    ("smart_quotes", '{“verdict”: “pass”, “overall_score”: 4}', "verdict"),
    ("prose_and_fence", 'My verdict:\n```json\n{"verdict": "pass", "note": "good {nested} braces"}\n```\nDone.', "verdict"),
    ("nested_object", '{"criteria": {"correctness": {"score": 5}}, "verdict": "pass"}', "verdict"),
]

FAIL_SAMPLES = [
    ("not_json", "The output was pretty good, I'd say 4 out of 5."),
    ("empty", ""),
]


def main() -> int:
    ok = 0
    total = 0
    for name, raw, key in MESSY_SAMPLES:
        total += 1
        obj, strat = parse_json_best_effort(raw)
        passed = isinstance(obj, dict) and key in obj
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:16} strategy={strat}")
        ok += int(passed)
    print("  --- expected-to-fail samples ---")
    for name, raw in FAIL_SAMPLES:
        total += 1
        obj, strat = parse_json_best_effort(raw)
        passed = obj is None
        print(f"  [{'PASS' if passed else 'FAIL'}] {name:16} strategy={strat} (correctly unparseable)")
        ok += int(passed)
    print(f"\n{ok}/{total} parser tests passed")
    return 0 if ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
