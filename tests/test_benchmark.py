"""Benchmark tests — `jarvis --benchmark-models` with everything mocked.

No real Ollama server is needed: requests.get/post are monkeypatched so
the benchmark runs against synthetic streamed responses whose timing
metrics are deterministic per model.
"""

import json

import pytest

import jarvis_cli.benchmark as bm
from jarvis_cli.benchmark import (
    BENCHMARK_QUESTIONS,
    GROUNDING_FACTS,
    aggregate,
    conciseness_score,
    grounding_score,
    recommend,
    relevance_score,
    run_benchmark,
    run_single,
)

# Per-model synthetic metrics: load (ns) and target tokens/sec.
LOADS_NS = {"qwen3:8b": 20_000_000_000, "qwen3:1.7b": 8_000_000_000,
            "llama3.2:3b": 10_000_000_000}
TOKPS = {"qwen3:8b": 3.0, "qwen3:1.7b": 7.0, "llama3.2:3b": 5.5}

# Synthetic answers per model. qwen3:8b is fully grounded on the
# current-info question; qwen3:1.7b invents a name; llama3.2:3b omits
# one fact (0.75 grounding) so qwen3:8b is the only 1.00-quality model
# and must win the recommendation despite being slower.
ANSWERS = {
    "qwen3:8b": {
        "hi": "Hello.",
        "Hello JARVIS.": "Hello, Sir.",
        "How are you?": "I am well, thank you.",
        "What is Python?": "Python is a high-level programming language.",
        "Explain a linked list.": "A linked list is a linear data structure made of nodes.",
        "Who is the current Chief Minister of Andhra Pradesh?":
            "The current Chief Minister of Andhra Pradesh is Nara "
            "Chandrababu Naidu, in office since June 2024.",
        "What can you do?": "I can open apps, search the web, and answer questions.",
    },
    "qwen3:1.7b": {
        "hi": "Hello.",
        "Hello JARVIS.": "Hello there.",
        "How are you?": "Fine.",
        "What is Python?": "Python is a language.",
        "Explain a linked list.": "A linked list has nodes.",
        "Who is the current Chief Minister of Andhra Pradesh?":
            "The current Chief Minister of Andhra Pradesh is K. Rosaiah.",
        "What can you do?": "I can help you.",
    },
    "llama3.2:3b": {
        "hi": "Hello.",
        "Hello JARVIS.": "Hello, Sir.",
        "How are you?": "All good.",
        "What is Python?": "Python is a general-purpose programming language.",
        "Explain a linked list.": "A linked list is a sequence of nodes.",
        "Who is the current Chief Minister of Andhra Pradesh?":
            "Naidu is the Chief Minister of Andhra Pradesh since June 2024.",
        "What can you do?": "I can open apps and answer questions.",
    },
}


class FakeStreamResponse:
    """Stand-in for a streaming requests.Response."""

    def __init__(self, lines, status_code=200):
        self._lines = lines
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    def iter_lines(self, decode_unicode=False):
        return iter(self._lines)

    def json(self):
        return {}

    def close(self):
        pass


class FakeTagsResponse:
    status_code = 200

    def __init__(self, names):
        self.names = names

    def raise_for_status(self):
        pass

    def json(self):
        return {"models": [{"name": n} for n in self.names]}


def _stream_lines(model, question):
    """Synthetic NDJSON stream for one question, with deterministic
    Ollama timing metrics in the done chunk."""
    text = ANSWERS[model][question]
    words = text.split()
    lines = [
        json.dumps({"message": {"content": w + " "}, "done": False})
        for w in words
    ]
    done = {
        "message": {"content": ""},
        "done": True,
        "load_duration": LOADS_NS[model],
        "prompt_eval_count": 15,
        "prompt_eval_duration": 1_000_000_000,
        "eval_count": len(words),
        "eval_duration": int(len(words) / TOKPS[model] * 1e9),
    }
    lines.append(json.dumps(done))
    return lines


def _fake_post(monkeypatch):
    def post(url, json=None, **kwargs):
        model = json["model"]
        question = json["messages"][-1]["content"]
        return FakeStreamResponse(_stream_lines(model, question))

    monkeypatch.setattr(bm.requests, "post", post)


def _fake_tags(monkeypatch, names=None):
    names = names or ["qwen3:8b", "qwen3:1.7b", "llama3.2:3b"]
    monkeypatch.setattr(
        bm.requests, "get",
        lambda *a, **k: FakeTagsResponse(names),
    )


# ── Scoring helpers ───────────────────────────────────────────

def test_grounding_score_uses_supplied_facts():
    row = {
        "context": bm.MOCK_WEB_CONTEXT,
        "answer": "Nara Chandrababu Naidu is the Chief Minister of "
                  "Andhra Pradesh since June 2024.",
    }
    assert grounding_score(row) == 1.0


def test_grounding_score_punishes_invented_facts():
    row = {
        "context": bm.MOCK_WEB_CONTEXT,
        "answer": "The current Chief Minister of Andhra Pradesh is K. Rosaiah.",
    }
    assert grounding_score(row) < 1.0


def test_grounding_score_none_without_context():
    assert grounding_score({"answer": "anything"}) is None


def test_relevance_knowledge_questions():
    assert relevance_score(
        {"category": "knowledge", "question": "What is Python?",
         "answer": "Python is a language."}
    ) == 1.0
    assert relevance_score(
        {"category": "knowledge", "question": "What is Python?",
         "answer": "Jupiter is a planet."}
    ) == 0.0


def test_conciseness_penalizes_rambling():
    assert conciseness_score({"chars": 100}) == 1.0
    assert conciseness_score({"chars": 0}) == 0.0
    assert conciseness_score({"chars": 5000}) == 0.0
    assert 0.0 < conciseness_score({"chars": 400}) < 1.0


# ── Aggregation ───────────────────────────────────────────────

def test_aggregate_computes_tokens_per_sec():
    # eval_ns is already in SECONDS (run_single converts ns -> s).
    rows = [
        {"tokens": 10, "eval_ns": 2.0, "chars": 40, "ttft": 1.0,
         "total": 5.0, "prompt_eval": 0.5, "context": None,
         "question": "q", "category": "conversation", "answer": "x" * 40},
        {"tokens": 20, "eval_ns": 3.0, "chars": 80, "ttft": 2.0,
         "total": 6.0, "prompt_eval": 0.5, "context": None,
         "question": "q2", "category": "conversation", "answer": "y" * 80},
    ]
    s = aggregate("m", load=4.0, rows=rows)
    assert s["tokens_per_sec"] == 30 / 5.0  # 30 tokens / 5s
    assert s["load"] == 4.0
    assert s["tokens"] == 30
    assert s["ttft"] == 1.5
    assert s["chars"] == 60
    assert 0.0 < s["quality"] <= 1.0


# ── Recommendation ────────────────────────────────────────────

def _summary(model, quality, tokps):
    return {"model": model, "quality": quality,
            "tokens_per_sec": tokps, "grounding": None,
            "total": 10.0, "ttft": 2.0, "chars": 100}


def test_recommend_prefers_quality_over_speed():
    """The fast model is excluded when its quality lags too far — even
    though it is the fastest (the FINAL RULE: accuracy beats speed)."""
    results = [
        _summary("qwen3:1.7b", 0.50, 20.0),
        _summary("qwen3:8b", 0.95, 3.0),
    ]
    rec = recommend(results)
    assert rec["model"] == "qwen3:8b"


def test_recommend_ties_break_by_speed():
    results = [
        _summary("qwen3:8b", 0.90, 3.0),
        _summary("llama3.2:3b", 0.90, 5.5),
    ]
    rec = recommend(results)
    assert rec["model"] == "llama3.2:3b"


def test_recommend_empty_returns_none():
    assert recommend([]) is None


# ── run_single metric parsing ─────────────────────────────────

def test_run_single_parses_ollama_metrics(monkeypatch):
    monkeypatch.setattr(
        bm.requests, "post",
        lambda url, json=None, **k: FakeStreamResponse(
            _stream_lines("qwen3:8b", "What is Python?")
        ),
    )
    client = type("C", (), {
        "base_url": "http://fake",
        "timeout": 30,
        "_build_messages": lambda self, q, m, context=None: [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": q},
        ],
        "_build_payload": lambda self, msgs, stream: {
            "model": "qwen3:8b", "messages": msgs, "stream": stream,
        },
    })()
    row = run_single(client, "What is Python?")
    assert row["answer"].strip().startswith("Python")
    assert row["load"] == 20.0
    assert row["prompt_tokens"] == 15
    assert row["prompt_eval"] == 1.0
    assert row["tokens"] > 0
    assert row["tokens_per_sec"] == pytest.approx(3.0, rel=0.2)


# ── End-to-end (fully mocked) ─────────────────────────────────

def test_run_benchmark_full_flow(monkeypatch, capsys):
    _fake_tags(monkeypatch)
    _fake_post(monkeypatch)
    assert run_benchmark() == 0
    out = capsys.readouterr().out

    assert "MODEL BENCHMARK" in out
    assert "BENCHMARK SUMMARY" in out
    assert "qwen3:8b" in out and "qwen3:1.7b" in out and "llama3.2:3b" in out
    # Mocked load times appear in the detail/table.
    assert "20.0" in out  # qwen3:8b load (from mocked load_duration)
    # Every model answers every shared question.
    assert "Who is the current Chief Minister of Andhra Pradesh?" in out
    # The grounded answer uses the mocked context.
    assert "Naidu" in out
    # Recommendation is deterministic: quality beats speed.
    assert "RECOMMENDED MODEL: qwen3:8b" in out


def test_run_benchmark_ollama_down(monkeypatch, capsys):
    def boom(*a, **k):
        raise bm.requests.ConnectionError("down")

    monkeypatch.setattr(bm.requests, "get", boom)
    monkeypatch.setattr(bm.requests, "post", boom)
    assert run_benchmark() == 1
    assert "Cannot reach Ollama" in capsys.readouterr().out


def test_run_benchmark_skips_uninstalled_models(monkeypatch, capsys):
    _fake_tags(monkeypatch, names=["qwen3:8b"])
    _fake_post(monkeypatch)
    assert run_benchmark() == 0
    out = capsys.readouterr().out
    assert "[skip] qwen3:1.7b not installed" in out
    assert "[skip] llama3.2:3b not installed" in out
    assert "RECOMMENDED MODEL: qwen3:8b" in out


def test_benchmark_questions_share_shape():
    """Every model must answer the SAME questions (incl. categories A-D)."""
    categories = {q.category for q in BENCHMARK_QUESTIONS}
    assert categories == {"conversation", "knowledge", "current", "command"}
    assert len(BENCHMARK_QUESTIONS) >= 6
    # The current-information question carries the mocked context.
    current = [q for q in BENCHMARK_QUESTIONS if q.category == "current"]
    assert current and current[0].context == bm.MOCK_WEB_CONTEXT
    # All grounding facts exist in the mocked context.
    ctx_lower = bm.MOCK_WEB_CONTEXT.lower()
    for fact in GROUNDING_FACTS:
        assert fact in ctx_lower


def test_default_models_include_candidates():
    assert bm.DEFAULT_MODELS == ["qwen3:8b", "qwen3:1.7b", "llama3.2:3b"]
