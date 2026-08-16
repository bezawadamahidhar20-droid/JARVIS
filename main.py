"""
main.py — JARVIS AI Assistant

Flow:
  Microphone → VAD → Faster-Whisper → IntentRouter → [Command | AI] → Piper TTS

Architecture:
  initialize()
    ├── validate_environment()   — Python, mic, Ollama (never crashes)
    ├── load_models()            — Whisper + Piper voice, loaded ONCE
    ├── initialize_audio()       — VAD calibration, mic/STT/TTS ready
    └── initialize_router()      — intent router + command registry
  start_listening_loop()         — listen → transcribe → route → respond

Every individual failure is contained: JARVIS only stops on exit or an
unrecoverable condition.

The AI brain is provider-agnostic (brain/llm.py). If Ollama is
unavailable JARVIS still runs — local commands keep working and a
friendly "offline" message is spoken for AI questions.
"""

import inspect
import re
import sys
import time
import threading

# ── Config ────────────────────────────────────────────────────
from config import jarvis_config, ollama_config, tts_config

# ── Engine ────────────────────────────────────────────────────
from engine.microphone import MicrophoneManager
from engine.stt import STTEngine
from engine.tts import TTSEngine

# ── Brain ─────────────────────────────────────────────────────
from brain.memory import ConversationMemory
from brain.llm import create_provider
from brain.router import IntentRouter, Intent, validate_input
from brain.search import (
    build_search_query,
    filter_and_rank,
    format_results_for_llm,
)

# ── Commands ──────────────────────────────────────────────────
from commands.registry import CommandRegistry, PendingConfirmation

# ── Utils ─────────────────────────────────────────────────────
from utils.logger import get_logger, set_debug

logger = get_logger("main")

# ── Error messages JARVIS speaks ──────────────────────────────
ERROR_OLLAMA_DOWN = (
    f"I'm sorry {jarvis_config.OWNER}, my AI systems are "
    "currently offline. Please ensure Ollama is running."
)
ERROR_AI_FAILED = (
    "I encountered an error processing your request. "
    "Please try again."
)
ERROR_MEMORY_CLEARED = (
    "Memory cleared. I have forgotten our previous "
    "conversation and we are starting fresh."
)
ERROR_INPUT_TOO_LONG = (
    "I'm sorry, that message was too long. "
    f"Please keep your request under "
    f"{jarvis_config.MAX_INPUT_CHARS} characters."
)
ERROR_CONFIRMATION_TIMED_OUT = (
    f"I'm sorry {jarvis_config.OWNER}, that action timed out "
    "and has been cancelled."
)
ERROR_COMMAND_FAILED = (
    "Sorry, I couldn't complete that command."
)
# Spoken when a current-information question cannot be verified. JARVIS
# NEVER falls back to a potentially stale LLM answer for these.
ERROR_CANNOT_VERIFY = (
    f"I'm sorry {jarvis_config.OWNER}, I couldn't verify the latest "
    "information right now."
)

# Confirmation answers for CONFIRM-permission commands (shutdown etc).
_CONFIRM_YES_RE = re.compile(
    r"^(yes|yeah|yep|sure|go ahead|please do|do it|okay|ok|confirm|"
    r"proceed|affirmative)\b",
    re.IGNORECASE,
)
_CONFIRM_NO_RE = re.compile(
    r"^(no|nope|nah|cancel|cancel it|don't|dont|stop|abort|"
    r"never mind|forget it)\b",
    re.IGNORECASE,
)


class JARVIS:
    """
    Main JARVIS orchestrator.
    Wires all components together into a single pipeline.
    """

    def __init__(
        self,
        text_mode: bool = False,
        debug: bool = False,
        benchmark: bool = False,
        components: dict | None = None,
    ):
        mode = "TEXT MODE" if text_mode else "VOICE MODE"
        print("\n=============================================")
        print("              JARVIS ONLINE")
        print(f"              {mode}")
        print("=============================================")

        if debug:
            set_debug(True)

        self.text_mode = text_mode
        self.debug = debug
        self.benchmark = benchmark

        # Allow tests to inject fakes for every subsystem.
        components = components or {}

        # Audio pipeline
        self.mic = components.get("mic") or MicrophoneManager()
        self.stt = components.get("stt") or STTEngine(self.mic)
        self.tts = components.get("tts") or TTSEngine(rate=tts_config.RATE)

        # AI brain (provider-agnostic). NOTE: `or` would be wrong here —
        # an empty ConversationMemory is falsy (it defines __len__).
        memory_component = components.get("memory")
        self.memory = (
            memory_component if memory_component is not None
            else ConversationMemory()
        )
        self.provider = components.get("provider")
        if self.provider is None:
            try:
                self.provider = create_provider()
            except Exception as e:
                logger.warning(
                    f"AI provider unavailable at startup: {e}"
                )
                self.provider = None

        # Web search (current-information answers)
        if "search" in components:
            self.search = components["search"]
        else:
            try:
                from brain.search import create_search_provider

                self.search = create_search_provider()
            except Exception as e:
                logger.warning(f"Web search unavailable: {e}")
                self.search = None

        # Routing and commands
        self.router = components.get("router") or IntentRouter()
        self.commands = components.get("commands") or CommandRegistry()

        # Pending CONFIRM-permission command (shutdown/restart/sleep).
        # PendingConfirmation enforces an expiry window so a stale "yes"
        # can never execute a command; None when idle.
        self._pending: PendingConfirmation | None = None

        self.running = False

        # Active model mode (fast/quality). Starts from .env and can be
        # changed at runtime with a voice command ("switch to fast
        # mode") without restarting JARVIS.
        self.model_mode = (jarvis_config.MODEL_MODE or "quality").strip().lower()

        # Optional rich terminal dashboard (issue: polished real-time
        # interface). Only created when rich is importable.
        self.ui = None
        try:
            from utils.terminal_ui import TerminalUI

            self.ui = TerminalUI(debug=debug)
        except Exception as e:
            logger.debug(f"Terminal dashboard unavailable: {e}")
        self.ollama_ok = self._check_provider()
        self._timings: list[dict] = []
        # Warm-up coordination: the background load must not overlap
        # with the first real question (that would queue TWO requests
        # on Ollama and double the cold-start wait).
        self._warmup_thread: threading.Thread | None = None
        self._warmup_done = threading.Event()
        self._start_warmup()
        logger.info("JARVIS initialized successfully.")

    # ── Provider helpers ──────────────────────────────────────

    def _check_provider(self) -> bool:
        """Ping the AI provider without crashing anything."""
        if self.provider is None:
            return False
        try:
            return bool(self.provider.is_available())
        except Exception as e:
            logger.warning(f"Provider check failed: {e}")
            return False

    def _start_warmup(self) -> None:
        """Pre-load the configured model in the background at startup."""
        self._warmup_model()

    def _warmup_model(self) -> None:
        """
        Pre-load the provider's current model in a background thread so
        the first real question has no cold-start delay.

        Used at startup and after a runtime model-mode switch. A fresh
        event is created per load so a completed earlier warm-up never
        makes the next one return immediately.
        """
        if not jarvis_config.ENABLE_WARMUP:
            logger.info("Warm-up disabled (ENABLE_WARMUP=false).")
            self._warmup_done.set()
            return
        if not self.ollama_ok or self.provider is None:
            logger.info("AI not reachable; skipping warm-up.")
            self._warmup_done.set()
            return
        warmup = getattr(self.provider, "warmup", None)
        if warmup is None:
            self._warmup_done.set()
            return

        # Fresh event: the old one may already be set from an earlier
        # load (startup warm-up or a previous mode switch).
        self._warmup_done = threading.Event()

        def _run_warmup() -> None:
            try:
                warmup()
            finally:
                self._warmup_done.set()

        self._warmup_thread = threading.Thread(
            target=_run_warmup,
            name="jarvis-warmup",
            daemon=True,
        )
        self._warmup_thread.start()

    def _wait_for_warmup(self, timeout: float = 120.0) -> None:
        """
        If the model is still loading in the background, wait for it
        before firing the first real request. Without this, the warm-up
        and the answer would both queue on Ollama and the cold start
        would cost two overlapping loads instead of one.
        """
        if self._warmup_thread is not None and self._warmup_thread.is_alive():
            logger.debug("Waiting for model warm-up to finish...")
            self._warmup_done.wait(timeout=timeout)

    # ── Initialization ────────────────────────────────────────

    def initialize(self) -> None:
        """Run the full startup sequence in order."""
        self.validate_environment()
        self.load_models()
        self.initialize_audio()
        self.initialize_router()

    def validate_environment(self) -> None:
        """Check prerequisites. Never raises — reports problems only."""
        print("[+] Configuration loaded")

        if sys.version_info >= (3, 10):
            logger.debug(f"Python {sys.version_info.major}."
                         f"{sys.version_info.minor} OK")
        else:
            logger.error("Python 3.10+ required.")

        if self.mic.is_available():
            logger.info(f"Microphone: {self.mic.describe()}")
            self._ui_component("Microphone", "ok", self.mic.describe())
        else:
            logger.warning(
                "Microphone unavailable — commands and text mode "
                "still work."
            )
            self._ui_component("Microphone", "off", "unavailable")

        if self.ollama_ok:
            logger.info(f"AI engine: {self.provider.describe()}")
            print("[+] AI engine ready")
            self._ui_component("Ollama", "ok", self.provider.describe())
        else:
            print(
                "[!] Ollama unavailable — local command mode "
                "remains available."
            )
            logger.warning(
                "Ollama not reachable; conversational answers will "
                "be unavailable until it starts."
            )
            self._ui_component("Ollama", "off", "not reachable")

    @staticmethod
    def _describe(obj, default: str = "ready") -> str:
        """Call ``obj.describe()`` defensively (fakes may lack it)."""
        fn = getattr(obj, "describe", None)
        if fn is None:
            return default
        try:
            return str(fn())
        except Exception:
            return default

    def load_models(self) -> None:
        """Load Whisper + Piper voice models once at startup."""
        if self.stt.load():
            print("[+] Speech recognition ready")
            self._ui_component("Whisper", "ok", self._describe(self.stt))
        else:
            print(
                "[!] Speech recognition unavailable — text mode "
                "and commands still work."
            )
            self._ui_component("Whisper", "off", "unavailable")

        if self.tts.load():
            print("[+] Voice engine ready")
            self._ui_component(
                "TTS", "ok",
                getattr(self.tts, "engine_name", "tts"),
            )
        else:
            print(
                "[!] Voice engine unavailable — replies will be "
                "printed to the console instead."
            )
            self._ui_component("TTS", "off", "unavailable")

    def initialize_audio(self) -> None:
        """Calibrate the adaptive VAD against ambient noise."""
        calibrate = getattr(self.stt, "vad", None)
        if calibrate is not None and hasattr(calibrate, "calibrate"):
            calibrate.calibrate()
        logger.debug("Audio pipeline ready.")

    def initialize_router(self) -> None:
        """Wire up routing + commands."""
        print("[+] Command router ready")
        self._ui_component("Router", "ok", "ready")
        logger.debug("Router + command registry ready.")

    # ── Terminal dashboard helpers ────────────────────────────

    def _ui_component(self, name: str, status: str, detail: str) -> None:
        """Update a component row on the terminal dashboard (no-op when
        the dashboard is not active)."""
        if self.ui is not None:
            try:
                self.ui.set_component(name, status, detail)
            except Exception:
                pass

    def _ui_state(self, state: str, meta: str = "") -> None:
        """Update the dashboard state indicator (no-op when inactive)."""
        if self.ui is not None:
            try:
                self.ui.set_state(state, meta)
            except Exception:
                pass

    # ── Speech helpers ────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak text via TTS (non-blocking)."""
        if text and text.strip():
            self._ui_state("speaking")
            self.tts.speak(text)

    def speak_blocking(self, text: str) -> None:
        """Speak text and block until it finishes (farewells only)."""
        if text and text.strip():
            self.tts.speak_blocking(text)

    def listen(self):
        """Listen for user speech."""
        return self.stt.listen()

    def greet(self) -> None:
        """Startup greeting."""
        if self.text_mode:
            greeting = (
                f"Hello {jarvis_config.OWNER}. "
                "JARVIS online in text mode."
            )
        else:
            greeting = (
                f"Good day, {jarvis_config.OWNER}. "
                "I am JARVIS, at your service. "
                "How can I help you today?"
            )
        self.speak(greeting)

    # ── Handlers ──────────────────────────────────────────────

    def handle_ai_question(self, user_input: str) -> None:
        """
        Send question to the AI provider.
        Updates memory before and after for context tracking.
        """
        if self.provider is None or not self.ollama_ok:
            logger.error("AI provider not available.")
            self.speak(ERROR_OLLAMA_DOWN)
            return

        # Cold-start: never race the background model load.
        self._wait_for_warmup()

        # Add to memory BEFORE asking
        self.memory.add_user_message(user_input)

        print("[Thinking...]")
        t0 = time.perf_counter()

        if self.text_mode:
            response = self.provider.ask(user_input, self.memory)
        else:
            # Stream: each finished sentence is spoken immediately
            # while the rest of the answer is still generating.
            response = self.provider.ask_stream(
                user_input,
                self.memory,
                on_sentence=self.tts.speak,
            )

        logger.debug(
            f"[timing] interaction {(time.perf_counter() - t0):.1f}s"
        )

        if response:
            # Add response to memory for future context
            self.memory.add_assistant_message(response)
            if self.text_mode:
                self.speak(response)
        else:
            # Remove failed user message from memory
            if self.memory._messages:  # noqa: SLF001
                self.memory._messages.pop()  # noqa: SLF001
            self.speak(ERROR_AI_FAILED)

    # ── Web search (current information) ──────────────────────

    def handle_web_search(self, user_input: str) -> None:
        """
        Answer a current-information question from verified web results.

        Never silently falls back to a stale LLM answer: if the search
        is not configured, fails, or finds nothing usable, JARVIS says
        it could not verify the information (no hallucination).
        """
        if self.search is None or not self.search.is_configured():
            logger.warning(
                "Web search needed but not configured for: "
                f"'{user_input}'"
            )
            self.speak(ERROR_CANNOT_VERIFY)
            return

        # Cold-start: never race the background model load (the
        # grounded answer also needs the model).
        self._wait_for_warmup()

        print("[Searching...]")
        query = build_search_query(user_input)
        t_search = time.perf_counter()
        try:
            results = self.search.search(query, self.search.max_results)
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            self.speak(ERROR_CANNOT_VERIFY)
            return
        logger.debug(
            f"[timing] web search {(time.perf_counter() - t_search):.1f}s "
            f"({len(results)} raw results)"
        )

        results = filter_and_rank(results, query)
        if not results:
            print("[No reliable sources found]")
            self.speak(ERROR_CANNOT_VERIFY)
            return

        print(f"[Found {len(results)} sources]")

        # The LLM must answer ONLY from the retrieved information — we
        # never re-ask it the original question without the context.
        context = format_results_for_llm(results)

        if self.provider is None or not self.ollama_ok:
            # LLM unavailable: read the top verified snippet directly
            # (search results are verified info, not a hallucination).
            self._answer_from_snippet(results[0])
            return

        self.memory.add_user_message(user_input)
        print("[Thinking...]")
        t0 = time.perf_counter()

        if self.text_mode:
            response = self._provider_ask(
                user_input, self.memory, context=context
            )
        else:
            response = self._provider_ask(
                user_input,
                self.memory,
                context=context,
                stream=True,
                on_sentence=self.tts.speak,
            )

        logger.debug(
            f"[timing] web interaction {(time.perf_counter() - t0):.1f}s"
        )

        if response:
            self.memory.add_assistant_message(response)
            self._print_sources(results)
            if self.text_mode:
                self.speak(response)
        else:
            if self.memory._messages:  # noqa: SLF001
                self.memory._messages.pop()  # noqa: SLF001
            self.speak(ERROR_CANNOT_VERIFY)

    def _answer_from_snippet(self, result) -> None:
        """Speak a short verified answer from the top result when the
        LLM is unavailable. The snippet is retrieved evidence, so this
        is honest reporting, not hallucination."""
        snippet = " ".join((result.snippet or result.title).split())
        if not snippet:
            self.speak(ERROR_CANNOT_VERIFY)
            return
        self.speak(f"Based on my search: {snippet[:280]}")
        self._print_sources([result])

    def _print_sources(self, results, limit: int = 3) -> None:
        """Display sources on the terminal (never read aloud)."""
        print("[Sources]")
        for r in results[:limit]:
            if r.url:
                print(f"  - {r.title or r.source or r.url}")
                print(f"    {r.url}")

    def _provider_ask(
        self,
        user_input: str,
        memory,
        context: str | None = None,
        stream: bool = False,
        on_sentence=None,
    ):
        """
        Call the AI provider, passing search *context* only when the
        provider supports it (older/fake providers stay compatible).
        """
        fn = (
            self.provider.ask_stream if stream else self.provider.ask
        )
        kwargs = {}
        if stream:
            kwargs["on_sentence"] = on_sentence
        if context:
            try:
                if "context" in inspect.signature(fn).parameters:
                    kwargs["context"] = context
            except (TypeError, ValueError):
                pass
        return fn(user_input, memory, **kwargs)

    @staticmethod
    def _confirm_decision(text: str) -> str:
        """Classify a reply to a pending confirmation: yes / no / other."""
        t = (text or "").strip()
        if _CONFIRM_YES_RE.match(t):
            return "yes"
        if _CONFIRM_NO_RE.match(t):
            return "no"
        return "other"

    def handle_command(self, user_input: str) -> None:
        """
        Execute a system command.

        SAFE commands run immediately. CONFIRM commands (shutdown /
        restart / sleep) only store a PendingConfirmation and ask the
        user to confirm — they execute in process_input() after a
        clear "yes", and expire after CONFIRMATION_TIMEOUT seconds.
        """
        result = self.commands.execute_with_meta(user_input)
        if result is None:
            self.speak(
                "I understood that as a command, but I don't know "
                "how to do it."
            )
            return
        if result.needs_confirmation:
            self._pending = PendingConfirmation(result, user_input)
            self.speak(result.confirm_prompt)
            return
        if result.response:
            self.speak(result.response)

    # ── Runtime model-mode control (fast / quality) ───────────

    def switch_model_mode(self, mode: str) -> str:
        """
        Switch the active model mode at runtime (fast/quality).

        Resolves the target model from .env config, switches the
        provider's active model, and pre-loads it in the background so
        the next question has no cold-start delay.

        Returns the new model name, or '' when the switch could not be
        made (unknown mode, unconfigured model, provider limitation).
        """
        mode = (mode or "").strip().lower()
        if mode not in ("fast", "quality"):
            logger.warning(f"Unknown model mode: {mode!r}")
            return ""
        try:
            target = ollama_config.resolve_model(mode)
        except Exception as e:
            logger.warning(f"Cannot resolve model for mode {mode!r}: {e}")
            return ""
        if not target:
            return ""

        current = getattr(self.provider, "model", None)
        if current == target:
            self.model_mode = mode
            return target

        switcher = getattr(self.provider, "switch_model", None)
        if switcher is None:
            logger.warning(
                "Provider cannot switch models at runtime "
                f"({getattr(self.provider, 'name', '?')})."
            )
            return ""
        try:
            switcher(target)
        except Exception as e:
            logger.warning(f"Runtime model switch failed: {e}")
            return ""
        self.model_mode = mode
        self._warmup_model()
        return target

    def handle_model_mode(self, mode_request: str) -> None:
        """Handle a runtime model-mode request: "fast" | "quality" |
        "status". Routed here deterministically (no LLM round-trip)."""
        if mode_request == "status":
            mode = self.model_mode or (jarvis_config.MODEL_MODE or "quality")
            model = getattr(self.provider, "model", "") or ""
            if not model:
                model = ollama_config.resolve_model(mode)
            self.speak(f"I am running in {mode} mode with {model}.")
            return

        previous = self.model_mode
        target = self.switch_model_mode(mode_request)
        if not target:
            self.speak(
                f"I couldn't switch to {mode_request} mode. "
                "Check OLLAMA_FAST_MODEL and OLLAMA_QUALITY_MODEL "
                "in the env file."
            )
            return
        if previous == mode_request:
            self.speak(
                f"I'm already using {target} in {mode_request} mode."
            )
        else:
            self.speak(
                f"Switched to {mode_request} mode. Using {target}."
            )

    def process_input(self, user_input: str) -> bool:
        """
        Process one user input through the full pipeline.

        Returns:
            False = exit JARVIS
            True  = continue running
        """
        # Validate + normalize up front: empty/whitespace input is
        # ignored, and over-long input is rejected politely — it never
        # reaches a command handler or the AI provider.
        validated = validate_input(user_input)
        if not validated:
            if user_input and len(user_input.strip()) > jarvis_config.MAX_INPUT_CHARS:
                logger.warning("Over-long input rejected.")
                self.speak(ERROR_INPUT_TOO_LONG)
            return True
        user_input = validated

        print(f"\n[You] {user_input}")
        if self.ui is not None:
            try:
                self.ui.add_message("user", user_input)
            except Exception:
                pass

        # A CONFIRM command (shutdown/restart/sleep) is pending — the
        # next input must answer it before anything else is processed.
        if self._pending is not None:
            if self._pending.is_expired:
                # Stale confirmation: cancel it, never execute.
                logger.info("Pending confirmation expired; cancelling.")
                self._pending = None
                self.speak(ERROR_CONFIRMATION_TIMED_OUT)
            else:
                decision = self._confirm_decision(user_input)
                if decision == "yes":
                    # take() claims the action exactly once — a second
                    # "yes" cannot re-execute it.
                    claimed = self._pending.take()
                    self._pending = None
                    if claimed is not None:
                        result, original = claimed
                        try:
                            self.speak(result.execute(original))
                        except Exception as e:
                            logger.error(f"Confirmed command failed: {e}")
                            self.speak(ERROR_COMMAND_FAILED)
                    return True
                if decision == "no":
                    self._pending = None
                    self.speak(
                        f"Understood, {jarvis_config.OWNER}. I won't."
                    )
                    return True
                # Not a clear yes/no — drop the pending action and treat
                # this as a fresh request.
                self._pending = None

        # Route the input to correct handler
        intent, cleaned = self.router.route(user_input)

        if intent == Intent.FAST_RESPONSE:
            # Instant canned reply — no AI round-trip needed.
            self.speak(cleaned)
            return True

        if intent == Intent.EXIT:
            self.speak_blocking(
                f"Goodbye, {jarvis_config.OWNER}. "
                "JARVIS shutting down."
            )
            return False

        elif intent == Intent.CLEAR_MEMORY:
            self.memory.clear()
            self.speak(ERROR_MEMORY_CLEARED)

        elif intent == Intent.COMMAND:
            self._ui_state("thinking", "executing command")
            self.handle_command(cleaned)

        elif intent == Intent.STOP_SPEECH:
            # Interrupt TTS without routing through the AI.
            self.tts.stop()

        elif intent == Intent.MODEL_MODE:
            # Switch fast/quality model at runtime (no LLM, no restart).
            self._ui_state("thinking", "switching model mode")
            self.handle_model_mode(cleaned)

        elif intent == Intent.WEB_SEARCH:
            # Current-information question — answer from fresh search
            # results, never from the model's stale training data.
            self._ui_state("thinking", "searching the web")
            self.handle_web_search(cleaned or user_input)

        else:
            # AI_QUESTION or UNKNOWN — send to the AI provider.
            self._ui_state("thinking", "consulting the brain")
            self.handle_ai_question(user_input)

        self._ui_state("idle")
        return True

    # ── Benchmarking ──────────────────────────────────────────

    def _record_turn(self, timings: dict) -> None:
        if self.benchmark:
            self._timings.append(timings)

    def _print_benchmark(self) -> None:
        """Summarise per-stage latency across the session."""
        if not self._timings:
            print("\n[Benchmark] No interactions recorded.")
            return
        stages = ("listen", "process", "speak")
        print("\n=========== JARVIS BENCHMARK ===========")
        print(f"  Turns: {len(self._timings)}")
        for stage in stages:
            values = [
                t.get(stage, 0.0) for t in self._timings
                if t.get(stage) is not None
            ]
            if not values:
                continue
            avg = sum(values) / len(values)
            print(
                f"  {stage:<8} avg {avg*1000:6.0f} ms | "
                f"min {min(values)*1000:6.0f} ms | "
                f"max {max(values)*1000:6.0f} ms"
            )
        total = sum(
            t.get("listen", 0) + t.get("process", 0) + t.get("speak", 0)
            for t in self._timings
        )
        print(f"  Total session time: {total:.1f}s")
        print("=========================================")

    # ── Main loop ─────────────────────────────────────────────

    def start_listening_loop(self) -> None:
        """The listen → transcribe → route → respond loop."""
        print("\n=============================================")
        if self.text_mode:
            print("  JARVIS — TEXT MODE")
            print("  Type your message and press Enter.")
            print("  Type 'exit' to quit.")
        else:
            print(f"  {jarvis_config.NAME} is ready.")
            print("  Speak a command or question.")
            print("  Say 'goodbye' to exit.")
        print("=============================================\n")

        # Live terminal dashboard: colored, animated, state-aware.
        if self.ui is not None:
            try:
                self.ui.start()
                recorder = getattr(self.stt, "vad", None)
                if recorder is not None:
                    self.ui.set_recorder(recorder)
            except Exception as e:
                logger.warning(f"Terminal dashboard failed to start: {e}")
                self.ui = None

        consecutive_failures = 0
        MAX_FAILURES = 5

        while self.running:
            turn: dict = {}
            try:
                if self.text_mode:
                    user_input = input("\nYou: ").strip()
                    if not user_input:
                        continue
                else:
                    # Never listen while JARVIS is still speaking,
                    # or the mic will pick up JARVIS's own voice (echo).
                    t0 = time.perf_counter()
                    self.tts.wait()
                    turn["speak"] = time.perf_counter() - t0

                    self._ui_state("listening")
                    t0 = time.perf_counter()
                    user_input = self.listen()
                    turn["listen"] = time.perf_counter() - t0

                    if user_input is None:
                        consecutive_failures += 1
                        if consecutive_failures >= MAX_FAILURES:
                            logger.warning(
                                "Multiple listen failures. "
                                "Check microphone."
                            )
                            consecutive_failures = 0
                        continue

                consecutive_failures = 0

                t0 = time.perf_counter()
                should_continue = self.process_input(user_input)
                turn["process"] = time.perf_counter() - t0

                if not should_continue:
                    self.running = False
                    break

                self._record_turn(turn)
                logger.info(
                    f"[timing] total {turn['process']:.1f}s "
                    f"(process)"
                )
                time.sleep(0.1)

            except KeyboardInterrupt:
                print("\n[Interrupted]")
                self.speak_blocking(
                    f"Shutting down. "
                    f"Goodbye, {jarvis_config.OWNER}."
                )
                self.running = False
                break

            except Exception as e:
                # Log the message to console; trace only in debug mode.
                logger.error(f"Unexpected error in main loop: {e}")
                logger.debug("Traceback:", exc_info=True)
                time.sleep(1)
                continue

    def shutdown(self) -> None:
        """Release resources and exit cleanly."""
        self.running = False
        try:
            self.tts.stop()
        except Exception:
            pass
        try:
            self.stt.unload()
        except Exception:
            pass
        if self.ui is not None:
            try:
                self.ui.stop()
            except Exception:
                pass
            self.ui = None
        logger.info("JARVIS stopped.")
        print("\n[*] JARVIS offline. Goodbye.")
        if self.benchmark:
            self._print_benchmark()

    def run(self) -> None:
        """Full lifecycle: initialize → greet → listen → shutdown."""
        self.running = True
        self.initialize()
        self.greet()
        try:
            self.start_listening_loop()
        finally:
            self.shutdown()


def validate_configuration() -> list:
    """Audit config and log every problem. Returns the fatal ones."""
    from config import validate_config

    problems = validate_config()
    fatal = []
    for p in problems:
        if p["fatal"]:
            logger.error(f"Config error — {p['setting']}: {p['message']}")
            fatal.append(p)
        else:
            logger.warning(f"Config warning — {p['setting']}: {p['message']}")
    return fatal


def run_assistant(
    text_mode: bool = False,
    debug: bool = False,
    benchmark: bool = False,
) -> int:
    """Entry point used by both `python main.py` and `jarvis`."""
    # Fail early on genuinely invalid required configuration (missing
    # .env values use safe defaults, so only real mistakes surface).
    fatal = validate_configuration()
    if fatal:
        print("\nJARVIS cannot start — invalid configuration:")
        for p in fatal:
            print(f"  - {p['setting']}: {p['message']}")
        print("Fix the values in .env (see .env.example) and try again.\n")
        return 1

    jarvis = JARVIS(
        text_mode=text_mode,
        debug=debug,
        benchmark=benchmark,
    )
    jarvis.run()
    return 0


if __name__ == "__main__":
    argv = sys.argv[1:]
    run_assistant(
        text_mode=("--text" in argv or "--text-mode" in argv),
        debug=("--debug" in argv),
        benchmark=("--benchmark" in argv),
    )
