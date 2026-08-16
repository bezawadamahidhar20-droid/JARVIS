"""
jarvis_cli/benchmark.py — `jarvis --benchmark-models`

Benchmarks the candidate Ollama models (qwen3:8b, qwen3:1.7b,
llama3.2:3b) on the SAME questions so the comparison is apples-to-apples:

    A. simple conversation   ("Hello JARVIS.", "How are you?")
    B. general knowledge     ("What is Python?", "Explain a linked list.")
    C. current information   ("Who is the current Chief Minister of Andhra
                              Pradesh?") — answered from MOCKED Tavily
                              results, so repeated runs never consume
                              search API credits, and each model's
                              grounding can be checked deterministically
    D. command-style         ("What can you do?")

For every question it measures (from Ollama's own /api/chat response):

    - model load time       (load_duration)
    - prompt evaluation     (prompt_eval_duration + prompt_eval_count)
    - first-token latency   (client-side TTFT)
    - total generation      (wall clock)
    - tokens/sec            (eval_count / eval_duration)
    - generated tokens      (eval_count)
    - response length       (chars)

It then prints a summary table, a per-model quality assessment
(grounding + conciseness), and a final recommendation.

IMPORTANT: the default model is NEVER changed. The benchmark only
reports — switch the model yourself in .env (OLLAMA_MODEL, or
JARVIS_MODEL_MODE with OLLAMA_FAST_MODEL / OLLAMA_QUALITY_MODEL).
"""

import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import requests

from utils.logger import get_logger

logger = get_logger("benchmark")

# ── Models to compare ─────────────────────────────────────────
# Override with BENCHMARK_MODELS="qwen3:8b,qwen3:1.7b" in .env.
DEFAULT_MODELS = ["qwen3:8b", "qwen3:1.7b", "llama3.2:3b"]


def _benchmark_models() -> list[str]:
    """The models to benchmark (env override or the defaults)."""
    raw = os.getenv("BENCHMARK_MODELS", "")
    if raw.strip():
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(DEFAULT_MODELS)


# ── Mocked Tavily results (current-information question) ──────
# Fixed, verified facts — no live API call, no credits, and a
# deterministic grounding check for every model.
MOCK_WEB_CONTEXT = (
    "1. Nara Chandrababu Naidu — Wikipedia\n"
    "   Nara Chandrababu Naidu is the current Chief Minister of Andhra "
    "Pradesh, in office since 12 June 2024.\n"
    "   Source: https://en.wikipedia.org/wiki/Nara_Chandrababu_Naidu\n"
    "2. Andhra Pradesh — The Hindu\n"
    "   Chandrababu Naidu returned as Chief Minister of Andhra Pradesh "
    "in June 2024 after the state assembly elections.\n"
    "   Source: https://www.thehindu.com/news/national/andhra-pradesh/\n"
)

# Key facts the grounded answer MUST use (all present in MOCK_WEB_CONTEXT).
# A model that invents a different name/date fails the grounding test.
GROUNDING_FACTS = ("naidu", "chandrababu", "andhra pradesh", "2024")


@dataclass
class BenchQuestion:
    """One benchmark question shared by every model."""

    category: str
    question: str
    context: Optional[str] = None


BENCHMARK_QUESTIONS = [
    BenchQuestion("conversation", "Hello JARVIS."),
    BenchQuestion("conversation", "How are you?"),
    BenchQuestion("knowledge", "What is Python?"),
    BenchQuestion("knowledge", "Explain a linked list."),
    BenchQuestion(
        "current",
        "Who is the current Chief Minister of Andhra Pradesh?",
        context=MOCK_WEB_CONTEXT,
    ),
    BenchQuestion("command", "What can you do?"),
]


# ── Low-level measurement ─────────────────────────────────────

def _ns(value) -> float:
    """Convert a nanosecond count to seconds (0.0 on None/invalid)."""
    try:
        return float(value or 0) / 1e9
    except (TypeError, ValueError):
        return 0.0


def run_single(client, question: str, context: Optional[str] = None) -> dict:
    """Ask *question* through *client* (streaming) and return metrics.

    Uses the same message/payload builders as real JARVIS requests, so
    the numbers reflect production behavior (system prompt, num_predict,
    num_ctx, keep_alive, think:false). Raises on any failure — the
    caller reports it per model.
    """
    messages = client._build_messages(question, None, context=context)  # noqa: SLF001
    payload = client._build_payload(messages, stream=True)

    start = time.perf_counter()
    first_token_at = None
    full = ""
    done: dict = {}

    resp = requests.post(
        f"{client.base_url}/api/chat",
        json=payload,
        stream=True,
        timeout=client.timeout,
    )
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        chunk = (data.get("message") or {}).get("content", "") or ""
        if chunk:
            if first_token_at is None:
                first_token_at = time.perf_counter()
            full += chunk
        if data.get("done"):
            done = data
            break
    try:
        resp.close()
    except Exception:
        pass

    total = time.perf_counter() - start
    eval_ns = _ns(done.get("eval_duration"))
    eval_count = int(done.get("eval_count") or 0)

    return {
        "question": question,
        "context": context,
        "load": _ns(done.get("load_duration")),
        "prompt_eval": _ns(done.get("prompt_eval_duration")),
        "prompt_tokens": int(done.get("prompt_eval_count") or 0),
        "ttft": (first_token_at - start) if first_token_at is not None else total,
        "total": total,
        "eval_ns": eval_ns,
        "tokens": eval_count,
        "tokens_per_sec": (eval_count / eval_ns) if eval_ns > 0 else 0.0,
        "chars": len(full.strip()),
        "answer": full.strip(),
    }


# ── Quality scoring ───────────────────────────────────────────

def grounding_score(row: dict) -> float:
    """Fraction of the supplied verified facts the answer actually uses.

    1.0 = the model answered from the provided context; 0.0 = it
    invented facts. Returns None for questions without context.
    """
    if not row.get("context"):
        return None
    text = (row.get("answer") or "").lower()
    if not text:
        return 0.0
    hits = sum(1 for fact in GROUNDING_FACTS if fact in text)
    return hits / len(GROUNDING_FACTS)


def relevance_score(row: dict) -> float:
    """Light keyword relevance — does the answer stay on the question's
    topic? Only knowledge questions get a deterministic check; every
    other category just needs a non-empty answer."""
    q = (row.get("question") or "").lower()
    text = (row.get("answer") or "").lower()
    if not text:
        return 0.0
    if row.get("category") == "knowledge":
        if "linked list" in q and ("linked list" in text or "list" in text):
            return 1.0
        if "python" in q and "python" in text:
            return 1.0
        return 0.0
    return 1.0


def conciseness_score(row: dict, max_chars: int = 320) -> float:
    """1.0 for short, voice-friendly answers; long rambling answers are
    degraded linearly. Empty answers score 0."""
    chars = row.get("chars", 0)
    if chars <= 0:
        return 0.0
    if chars <= max_chars:
        return 1.0
    return max(0.0, 1.0 - (chars - max_chars) / 500.0)


def _row_quality(row: dict) -> float:
    """Overall quality of one answer: relevance + conciseness, plus
    grounding when the question carried verified context."""
    scores = [relevance_score(row), conciseness_score(row)]
    g = grounding_score(row)
    if g is not None:
        scores.append(g)
    return sum(scores) / len(scores)


# ── Aggregation ───────────────────────────────────────────────

def aggregate(model: str, load: float, rows: list[dict]) -> dict:
    """Collapse per-question rows into a per-model summary."""
    n = max(1, len(rows))
    total_tokens = sum(r["tokens"] for r in rows)
    total_eval_ns = sum(r["eval_ns"] for r in rows)
    grounded = [r for r in rows if r.get("context")]
    return {
        "model": model,
        "load": load,
        "ttft": sum(r["ttft"] for r in rows) / n,
        "total": sum(r["total"] for r in rows) / n,
        "prompt_eval": sum(r["prompt_eval"] for r in rows) / n,
        "tokens": total_tokens,
        "tokens_per_sec": (
            total_tokens / total_eval_ns if total_eval_ns > 0 else 0.0
        ),
        "chars": sum(r["chars"] for r in rows) / n,
        "quality": sum(_row_quality(r) for r in rows) / n,
        "grounding": (
            sum(grounding_score(r) for r in grounded) / len(grounded)
            if grounded else None
        ),
        "rows": rows,
    }


# ── Recommendation ────────────────────────────────────────────

# A model whose quality lags the best by more than this is never
# recommended, no matter how fast it is (accuracy beats speed).
QUALITY_TOLERANCE = 0.10


def recommend(results: list[dict]) -> Optional[dict]:
    """Pick the model to recommend from measured data.

    Rules:
      * quality first — models whose quality lags the best by more than
        QUALITY_TOLERANCE are excluded, however fast they are;
      * among acceptable models, prefer the best quality, breaking ties
        by tokens/sec (CPU suitability).
    """
    if not results:
        return None
    best_quality = max(r["quality"] for r in results)
    acceptable = [
        r for r in results if r["quality"] >= best_quality - QUALITY_TOLERANCE
    ]
    if not acceptable:
        acceptable = results
    acceptable.sort(
        key=lambda r: (r["quality"], r["tokens_per_sec"]), reverse=True
    )
    return acceptable[0]


# ── Printing ──────────────────────────────────────────────────

def print_table(summaries: list[dict]) -> None:
    """Print the MODEL / LOAD / TTFT / TOK/S / TOTAL table."""
    print("\n=============================================")
    print("             BENCHMARK SUMMARY")
    print("=============================================")
    print(
        f"  {'MODEL':<14} {'LOAD(s)':>8} {'TTFT(s)':>8} "
        f"{'TOK/S':>7} {'TOTAL(s)':>8}"
    )
    for s in summaries:
        print(
            f"  {s['model']:<14} {s['load']:>8.1f} {s['ttft']:>8.1f} "
            f"{s['tokens_per_sec']:>7.1f} {s['total']:>8.1f}"
        )
    print("=============================================")


def print_quality(summaries: list[dict]) -> None:
    """Per-model quality assessment (quality score + grounding)."""
    print("\n  QUALITY (1.0 = best) - grounding is how well the model "
          "used the supplied web context for the current-information "
          "question (1.0 = fully grounded, 0.0 = invented facts).")
    print(
        f"  {'MODEL':<14} {'QUALITY':>8} {'GROUNDING':>10} {'AVG CHARS':>10}"
    )
    for s in summaries:
        grounding = (
            f"{s['grounding']:.2f}" if s["grounding"] is not None else "n/a"
        )
        print(
            f"  {s['model']:<14} {s['quality']:>8.2f} {grounding:>10} "
            f"{s['chars']:>10.0f}"
        )


def print_recommendation(rec: dict, results: list[dict]) -> None:
    """Final recommendation with reasons (never changes any config)."""
    print("\n=============================================")
    print(f"  RECOMMENDED MODEL: {rec['model']}")
    print("  REASON:")
    print(f"    - Speed: {rec['tokens_per_sec']:.1f} tokens/sec "
          f"(avg total {rec['total']:.1f}s per answer, "
          f"first token {rec['ttft']:.1f}s)")
    print(f"    - Quality: {rec['quality']:.2f}/1.00 on the shared "
          "questions (relevance + conciseness)")
    if rec["grounding"] is not None:
        print(f"    - Grounding: {rec['grounding']:.2f}/1.00 - answered the "
              "web question from the supplied context")
    else:
        print("    - Grounding: n/a (no grounded question answered)")
    fastest = max(results, key=lambda r: r["tokens_per_sec"])
    print("    - CPU suitability: " + (
        f"fastest measured ({fastest['tokens_per_sec']:.1f} tok/s)"
        if fastest["model"] == rec["model"]
        else f"not the fastest ({fastest['tokens_per_sec']:.1f} tok/s on "
             f"{fastest['model']}), but its quality is acceptable"
    ))
    print("  NOTE: the default model was NOT changed. To switch, edit "
          "OLLAMA_MODEL in .env, or set JARVIS_MODEL_MODE=fast with "
          "OLLAMA_FAST_MODEL.")
    print("=============================================\n")


def print_detail(model: str, rows: list[dict]) -> None:
    """Per-question detail incl. the model's answer for manual review."""
    # ASCII separators only: box-drawing glyphs crash under the Windows
    # cp1252 console when stdout is piped (same issue as the doctor).
    print(f"\n  == {model} ==")
    for r in rows:
        answer = r["answer"] or "(no answer)"
        print(f"\n  [{r['category']}] {r['question']}")
        print(f"      Answer: {answer[:240]}")
        print(
            f"      load {r['load']:.1f}s | prompt eval {r['prompt_eval']:.1f}s "
            f"({r['prompt_tokens']} tok) | TTFT {r['ttft']:.1f}s | "
            f"total {r['total']:.1f}s | {r['tokens']} tok | "
            f"{r['tokens_per_sec']:.1f} tok/s | {r['chars']} chars"
        )


# ── Main entry ────────────────────────────────────────────────

def _benchmark_one(model: str, questions: list[BenchQuestion]) -> dict:
    """Warm up *model*, run every question, return the summary."""
    from brain.ollama_client import OllamaClient

    client = OllamaClient(model=model)

    # Warm-up: load the model so the timed questions measure *warm*
    # latency (like real JARVIS after its startup warm-up). The cold
    # load time is captured from this request's load_duration.
    warm = run_single(client, "hi")
    load = warm.get("load", 0.0)

    rows = []
    for q in questions:
        row = run_single(client, q.question, context=q.context)
        row["category"] = q.category
        row["model"] = model
        rows.append(row)

    return aggregate(model, load, rows)


def _installed_models(base_url: str) -> list[str]:
    """Names of models Ollama reports as installed ([] when unreachable)."""
    try:
        r = requests.get(f"{base_url}/api/tags", timeout=5)
        r.raise_for_status()
        return [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        return []


def _is_installed(model: str, installed: list[str]) -> bool:
    """Is *model* (e.g. "qwen3:1.7b") one of Ollama's installed models?

    Exact name match. A base-name prefix match is NOT enough — qwen3:8b
    being installed must not make qwen3:1.7b benchmarkable.
    """
    if model in installed:
        return True
    # Ollama reports a model pulled as plain "qwen3" (or "qwen3:latest")
    # under the bare name "qwen3" — tolerate that one case.
    return model.endswith(":latest") and model[: -len(":latest")] in installed


def run_benchmark(verbose: bool = False) -> int:
    """Run the full model benchmark. Returns 0 on success."""
    print("\n=============================================")
    print("      JARVIS MODEL BENCHMARK (CPU)")
    print("      Same questions for every model;")
    print("      web question uses MOCKED Tavily results.")
    print("=============================================")

    from config import ollama_config

    installed = _installed_models(ollama_config.BASE_URL)
    if not installed:
        print(f"\n[!] Cannot reach Ollama at {ollama_config.BASE_URL}. "
              "Start Ollama (ollama serve) and try again.")
        return 1

    models = _benchmark_models()
    summaries: list[dict] = []

    for model in models:
        if not _is_installed(model, installed):
            print(f"\n[skip] {model} not installed — run: ollama pull {model}")
            continue
        print(f"\n[benchmarking {model}]")
        try:
            summary = _benchmark_one(model, BENCHMARK_QUESTIONS)
        except Exception as e:
            logger.error(f"{model} benchmark failed: {e}")
            print(f"\n[!] {model} failed: {e}")
            continue
        summaries.append(summary)
        print_detail(model, summary["rows"])

    if not summaries:
        print("\nNo models could be benchmarked. Install at least one of: "
              + ", ".join(models))
        return 1

    print_table(summaries)
    print_quality(summaries)
    rec = recommend(summaries)
    if rec is not None:
        print_recommendation(rec, summaries)
    return 0
