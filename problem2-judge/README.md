# LLM-as-Judge Evaluation Pipeline (Groq-backed)

A judging pipeline that takes a test suite `{ input, system_prompt, model_output, expected_output?, criteria? }`,
produces a **structured, per-criterion verdict** from an LLM judge, and takes judge bias seriously:
each of the five named biases is **mitigated in code and measured with before/after numbers**.

Every LLM call — judge, generator, and JSON-repair — routes through **Groq's OpenAI-compatible
endpoint** (`https://api.groq.com/openai/v1`) using the `openai` Python SDK. No Anthropic/Google SDKs.

---

## Setup & run (< 10 minutes)

```bash
cd problem2-judge
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env          # then paste your GROQ_API_KEY into .env
#   get a key at https://console.groq.com/keys

# offline sanity check (no key needed) — proves the robust parser works:
python -m tests.test_parser

# full pipeline against REAL Groq calls:
python main.py --suite suites/test_suite.json --probes suites/adversarial_probes.json
```

Outputs land in `results/`:

| File | Contents |
|------|----------|
| `suite_report.json` | pass rate, mean per-criterion scores, per-case verdicts, parse-failure rate |
| `bias_report.json`  | before/after numbers for **all 5 biases** + position flip rate |
| `validation.json`   | test-retest consistency, adversarial-probe results, gold agreement / Cohen's κ |
| `ab_comparison.json`| config A vs B, pointwise + pairwise, **declared winner** |
| `usage.json`        | judge tokens / calls / estimated cost |

Every judge prompt + raw response is also saved to `logs/` (auditable/replayable):
`logs/judge_calls.jsonl`, `logs/calls/<id>.json`, and `logs/parse_failures.jsonl`.

Run a single stage with `--only score|bias|validate|ab|all`.

**Suite format:** JSON *or* YAML — the loader picks by extension (`suites/example_suite.yaml`
is a working YAML example).

**Active generation (`--generate`):** normally the suite supplies `model_output`. With
`--generate`, the pipeline calls the **generator model** to produce missing outputs
(and a second output from `system_prompt_b`, giving a real prompt-v1-vs-v2 A/B pair)
before judging — so judge and generator are exercised *independently*, not just
configured independently:

```bash
python main.py --generate --suite suites/example_suite.yaml   # generator writes answers, judge scores them
```

### Sample results & reproducibility (free-tier note)

The `results/*.json` checked in here are **real outputs** from an actual Groq run over
the 16-case suite (judge = `openai/gpt-oss-20b`, `reasoning_effort=low`). To regenerate
on your own key, just run the command above — the pipeline is fully deterministic in
structure.

**On free-tier keys:** a full run is ~250 calls and can hit the per-minute/daily token
rate limit. The client **retries with backoff automatically, so it never crashes** — a
throttled run just runs slower (or pauses). For a fast, clean run: use a **paid key**
(any OpenAI-compatible provider), or set `GROQ_MIN_CALL_INTERVAL=3` to pace under the
limit, or verify quickly with `--only score` first. A paid-key full run finishes in ~2 min.

---

## The two Groq model families (and why)

Configured via env vars, swappable independently without touching code:

| Role | Default model | Family | Org |
|------|---------------|--------|-----|
| **Judge** | `openai/gpt-oss-120b` | GPT-OSS (open-weight MoE) | OpenAI |
| **Generator** | `qwen/qwen3.6-27b` | Qwen | Alibaba |

> As of 2026-07, `llama-3.3-70b-versatile` was deprecated on Groq (2026-06-17), so the defaults
> use currently-supported families. Change `GROQ_JUDGE_MODEL` / `GROQ_GENERATOR_MODEL` in `.env` to
> any Groq model id; to flip roles just swap the two values.

**Why these two are a *meaningful* separation (self-enhancement mitigation):** they come from
different organizations with different pretraining corpora, tokenizers, and RLHF pipelines — **not
two sizes of one base model**. A judge cannot self-prefer outputs from a family it did not produce.
This is the deliberate self-enhancement-bias mitigation.

---

## Judging modes — and when each fits

The pipeline implements **pointwise** (primary) and **pairwise** (for A/B).

| Mode | What it does | When it fits |
|------|--------------|--------------|
| **Pointwise scoring** | scores one `model_output` against the rubric, per criterion | absolute quality gating, per-criterion diagnostics, tracking a metric over time |
| **Pairwise A-vs-B** | picks a winner between two outputs for the same input | "is v2 better than v1?" — relative preference is more stable than absolute scores; also where we measure **position bias** |
| **Reference-based** | (a variant of the above) uses `expected_output` as ground truth | when you have a gold answer; strengthens correctness/faithfulness judging |
| **Reference-free** | same, without `expected_output` | open-ended tasks with no single right answer |

Pointwise and pairwise are complementary: pointwise gives you *how good*, pairwise gives you *which
is better* with less score noise. The A/B deliverable uses pairwise (order-averaged) as the primary
signal and pointwise means as a tiebreaker.

---

## The rubric (structured, not a bare number)

Five weighted criteria, each with a definition and **1–5 score anchors** (`src/rubric.py`):

| Criterion | Weight | 1 ↔ 5 |
|-----------|:------:|-------|
| Correctness | 0.30 | central claim wrong ↔ fully correct under scrutiny |
| Faithfulness | 0.25 | fabricates / contradicts context ↔ every claim traceable to input |
| Completeness | 0.20 | answers a different question ↔ addresses every explicit part |
| Instruction-following | 0.15 | violates hard constraints ↔ honors every stated constraint |
| Tone & Safety | 0.10 | harmful / inappropriate ↔ appropriate and safe throughout |

Overall score = weight-normalized mean of present criteria. The judge returns per-criterion
`{score, rationale}` + `overall_score` + `verdict` as strict JSON.

---

## Bias handling — named, mitigated in code, measured

All five live in `src/bias_mitigations.py`; each returns concrete numbers written to
`results/bias_report.json`. "before" = mitigation OFF, "after" = mitigation ON, on the same probes.

| Bias | Mitigation (in code) | Measured |
|------|----------------------|----------|
| **Position (A/B order)** | run every pair in **both** orders; trust only order-agreed winners | **flip rate** = % of pairs whose winner changes when order swaps |
| **Verbosity / length** | length-control instruction in the prompt + a padded low-value probe | score gap (concise − padded) **before vs after** the instruction |
| **Self-enhancement** | judge family ≠ generator family (config-level assertion) | families-differ = PASS/FAIL + optional matched-quality gap probe |
| **Sycophancy / style** | force **per-criterion grounding** (quote the output) + a confidently-wrong probe | correctness score on the wrong probe **before vs after** grounding; `fooled_after` flag |
| **Score clustering** | few-shot anchors (a clear 1, 3, 5) calibrate the scale | score **spread** (std / range / distinct values) with vs without anchors |

Prompt-level toggles (`MitigationConfig` in `src/judge.py`) let each bias be scored with its
mitigation off, then on, producing the honest before/after deltas.

---

## Judge validation (`src/validate.py`)

Three artifacts (assignment asks for ≥1; we ship all three):

- **Test-retest consistency** — re-score each case N times at `temperature>0`; report the
  verdict flip rate and mean score std. (Reliability under noise.)
- **Adversarial probes** — a *verbose-but-wrong* and a *terse-but-correct* answer; report whether
  the judge was **fooled** (rewarded the wrong one / punished the right one).
- **Gold agreement (pluggable)** — agreement rate + **Cohen's κ** vs human labels, computed only
  over cases that carry `gold_label` (`pass`/`fail`) or `gold_overall`. No labels → reports
  `no_labels` instead of failing, so it's a no-op until you fill labels in.

---

## Cost tracking

Every Groq response includes `usage`; `src/groq_client.py` accumulates prompt/completion tokens,
call count, and an **estimated USD cost** using per-token Groq pricing (see `GROQ_PRICING_USD_PER_1M`).
Totals are printed at the end of each run and written to `results/usage.json`.

**Judge pricing (Groq, USD per 1M tokens — verify against <https://groq.com/pricing>):**

| Model | Input | Output |
|-------|------:|-------:|
| `openai/gpt-oss-120b` (default judge) | ~$0.15 | ~$0.75 |
| `qwen/qwen3.6-27b` (default generator) | ~$0.20 | ~$0.80 |

A full run over ~15 cases (score + bias + validate + A/B) is a few hundred judge calls at most —
typically a few cents. Rate limits on the free/dev tier are handled by **retry-with-backoff on 429s**
(honoring `Retry-After`) in the client wrapper, so a long run won't crash midway.

---

## Robust JSON parsing

Open-weight models emit malformed JSON more than closed frontier models. `src/parser.py` tries a
ladder: direct → strip ``` fences → extract first balanced `{…}` → light repairs (trailing commas,
single/smart quotes) → **LLM repair prompt** (ask the judge to reformat its own text). Every failure
is logged to `logs/parse_failures.jsonl`, and the parse strategy used is recorded per case so you can
see the real malformed-JSON rate. Validated offline by `python -m tests.test_parser` (12/12).

---

## Results — actual measured numbers (16-case suite)

Real numbers from the committed `results/*.json` (judge = `openai/gpt-oss-20b`, generator =
`qwen/qwen3.6-27b`). Reproduce with `python main.py`.

**Suite** — pass rate **0.812** · mean overall **4.325 / 5** · score std **1.405** (uses the full
scale, not clustered) · malformed-JSON parse-failure rate **0.0**.

**Bias (before → after mitigation):**

| Bias | Measure | Result |
|------|---------|--------|
| Position | pairwise flip rate (both orders, 16 pairs) | **0.0** — no order flips |
| Verbosity | concise − padded score gap | 0.0 → 0.0 — judge never over-scored padding |
| Self-enhancement | judge family vs generator family | **PASS** — gpt-oss ≠ qwen |
| Sycophancy | correctness on confidently-wrong probe | 1 → 1, **not fooled** |
| Score clustering | overall-score std (no anchors → anchors) | **1.291 → 1.344** — spread widened |

**Judge validation:** test-retest verdict flip rate **0.0** · adversarial **not fooled**
(verbose-but-wrong = **1.4**, terse-but-correct = **5.0**) · gold agreement **1.0**, Cohen's
**κ = 1.0** (16 labeled cases).

**A/B comparison:** winner **config_A_v1**, head-to-head **10–3** (3 ties, **0 order-inconsistent**
— consistent with the 0.0 position flip rate). Config B correctly won the 3 cases where its output
was genuinely better.

---

## Discussion — how biased before vs after, and would I let it gate a release?

**Before vs after (this judge).** The honest finding is that `gpt-oss-20b` is a *fairly
well-behaved* judge: the "before" numbers already show little position bias, no verbosity
preference, and it isn't fooled by confident-but-wrong answers — so the mitigations mostly
**confirm robustness** rather than swing a large delta. Only score-clustering shows a measurable
improvement (std 1.29 → 1.34). That is itself a legitimate result: *we probed for each bias and the
judge held up.* An open-weight judge is nonetheless generally the noisy instrument the problem warns
about — clustering, confident-tone susceptibility, order flips — which is exactly why the mitigations
and the before/after harness exist; on a weaker judge or harder probes the deltas would be larger.

**Would I let this judge gate a release?** Not unconditionally, and specifically *because* it's an
open-weight judge rather than a frontier closed model — open-weight judges are measurably noisier
(higher test-retest flip rate, more score clustering), so the risk of a wrong auto-decision is real.
I would allow it to gate only with guardrails:

- **Gate on pairwise A/B deltas, not absolute pointwise scores** — relative preference is more stable
  than the clustered absolute scale.
- **Require order-agreement** (drop order-inconsistent pairs) and a **minimum margin**, not a 51%
  squeaker.
- **Block on validation health**: refuse to gate if the test-retest flip rate or adversarial
  "fooled" rate exceeds a threshold — a judge that fails its own probes shouldn't gate anything.
- **Human-in-the-loop on the boundary**: auto-pass clear wins, auto-fail clear losses, route the
  ambiguous middle band to a human.
- **Track judge cost/tokens and drift** over time; re-run the adversarial probe set on every model
  swap (families change on Groq, as the llama-3.3 deprecation shows).

Bottom line: usable as a **fast pre-filter and regression signal**, and as a hard gate only for
high-confidence pairwise decisions with the guardrails above — with a frontier judge reserved for
the final call on anything close.

---

## To run it yourself / swap in your own data

Everything below already ships filled-in and working — only the API key is required to run.

1. **`GROQ_API_KEY`** in `.env` (copied from `.env.example`) — the only must-have.
2. **Test suite** — `suites/test_suite.json` ships with **16 varied, ready-to-run cases**. Swap in
   your own domain cases anytime: add `model_output_b` to a case to include it in the A/B comparison;
   add `gold_label`/`gold_overall` to enable Cohen's κ.
3. **Adversarial probes** — `suites/adversarial_probes.json` is pre-seeded with working
   verbosity / sycophancy / validation probes; edit to your domain if you like.
4. **(Optional) model choice** — defaults are `openai/gpt-oss-120b` (judge) and `qwen/qwen3.6-27b`
   (generator); the committed sample results used `gpt-oss-20b` to stay under free-tier limits. Swap
   via env vars; keep judge and generator in **different families** or the self-enhancement check
   reports FAIL.

Then `python main.py` regenerates every report.

## Project layout

```
problem2-judge/
├── main.py                     # CLI runner -> results/*.json
├── requirements.txt            # openai + python-dotenv (Groq via OpenAI SDK)
├── .env.example                # GROQ_API_KEY, GROQ_JUDGE_MODEL, GROQ_GENERATOR_MODEL
├── src/
│   ├── groq_client.py          # single Groq wrapper: retry/backoff on 429, usage/cost tracking
│   ├── rubric.py               # 5 weighted criteria + 1-5 anchors + mitigation instructions
│   ├── judge.py                # pointwise + pairwise prompts, few-shot anchors, JSON mode+fallback
│   ├── parser.py               # robust JSON parsing ladder + LLM repair
│   ├── bias_mitigations.py     # the 5 biases: mitigate + measure before/after
│   ├── aggregate.py            # suite report + two-config A/B winner
│   ├── validate.py             # test-retest, adversarial probes, Cohen's kappa
│   ├── generator.py            # active generator: writes model_output via the generator model
│   └── logging_utils.py        # auditable prompt/response logging
├── suites/
│   ├── test_suite.json         # 16 varied ready-to-run cases
│   ├── example_suite.yaml      # YAML example (same schema; shows --generate)
│   └── adversarial_probes.json # verbosity / sycophancy / validation probes
├── results/                    # committed real outputs (suite/bias/validation/ab/usage .json)
└── tests/test_parser.py        # offline parser tests (no API key)
```
