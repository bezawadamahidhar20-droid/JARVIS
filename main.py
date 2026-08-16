"""
main.py — JARVIS AI Assistant
 
[FIX M6] Added shutdown() method for proper async event loop cleanup.
[FIX m13] Added basic prompt injection defense in system prompt.
"""
 
import asyncio
import concurrent.futures
import re
import time
import threading
 
from config import jarvis_config, ollama_config
from brain.router import Intent, validate_input
from brain.exceptions import OllamaTimeoutError
from brain.search import (
    build_search_query,
    filter_and_rank,
    format_results_for_llm,
)
from commands.registry import PendingConfirmation
from di import (
    COMMANDS, MEMORY, MIC, PROVIDER, ROUTER,
    SEARCH, STT, TTS, UI, DependencyContainer, build_default_container,
)
from utils.logger import get_logger, set_debug
 
logger = get_logger("main")
 
ERROR_OLLAMA_DOWN = (
    f"I'm sorry {jarvis_config.OWNER}, my AI systems are "
    "currently offline. Please ensure Ollama is running."
)
ERROR_AI_FAILED = "I encountered an error processing your request. Please try again."
ERROR_MEMORY_CLEARED = "Memory cleared. Starting fresh."
ERROR_INPUT_TOO_LONG = (
    "I'm sorry, that message was too long. "
    f"Please keep your request under {jarvis_config.MAX_INPUT_CHARS} characters."
)
ERROR_CANNOT_VERIFY = (
    f"I'm sorry {jarvis_config.OWNER}, I couldn't verify the latest information right now."
)
 
_CONFIRM_YES_RE = re.compile(
    r"^(yes|yeah|yep|sure|go ahead|please do|do it|okay|ok|confirm|proceed|affirmative)\b",
    re.IGNORECASE,
)
_CONFIRM_NO_RE = re.compile(
    r"^(no|nope|nah|cancel|don't|dont|stop|abort|never mind|forget it)\b",
    re.IGNORECASE,
)
 
 
class JARVIS:
    """Main JARVIS orchestrator."""
 
    def __init__(
        self,
        text_mode: bool = False,
        debug: bool = False,
        benchmark: bool = False,
        components: dict | None = None,
        container: DependencyContainer | None = None,
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
 
        self.container = container or build_default_container(debug=debug)
        for key, value in (components or {}).items():
            self.container.override(key, value)
 
        self.mic = self.container.get(MIC)
        self.stt = self.container.get(STT)
        self.tts = self.container.get(TTS)
        self.memory = self.container.get(MEMORY)
        self.provider = self.container.get(PROVIDER)
        self.search = self.container.get(SEARCH)
        self.router = self.container.get(ROUTER)
        self.commands = self.container.get(COMMANDS)
 
        self._pending: PendingConfirmation | None = None
 
        # Async event loop on dedicated thread
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(
            target=self._loop.run_forever,
            name="jarvis-asyncio",
            daemon=True,
        )
        self._loop_thread.start()
 
        self.running = False
        # The concurrent.futures.Future wrapping the running listening
        # loop (set by start_listening_loop()), plus the asyncio task
        # itself (registered by astart_listening_loop()). shutdown()
        # cancels the task so it is never abandoned mid-flight.
        self._main_future = None
        self._main_task: asyncio.Task | None = None
        self.model_mode = (jarvis_config.MODEL_MODE or "quality").strip().lower()
 
        self.ui = None
        try:
            self.ui = self.container.get(UI)
        except Exception as e:
            logger.debug(f"Terminal dashboard unavailable: {e}")
 
        self.ollama_ok = self._check_provider()
        self._timings: list[dict] = []
        self._warmup_thread: threading.Thread | None = None
        self._warmup_done = threading.Event()
        self._start_warmup()
        
        logger.info("JARVIS initialized successfully.")
 
    def _run_coro(self, coro):
        """Run async coroutine on dedicated loop."""
        if self._loop.is_closed():
            return asyncio.run(coro)
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result()
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            return None
 
    def _check_provider(self) -> bool:
        if self.provider is None:
            return False
        try:
            return bool(self.provider.is_available())
        except Exception as e:
            logger.warning(f"Provider check failed: {e}")
            return False
 
    def _start_warmup(self) -> None:
        if jarvis_config.ENABLE_WARMUP and self.provider:
            self._warmup_thread = threading.Thread(
                target=self._warmup_model,
                name="jarvis-warmup",
                daemon=True,
            )
            self._warmup_thread.start()
        else:
            self._warmup_done.set()
 
    def _warmup_model(self) -> None:
        try:
            if hasattr(self.provider, "warmup"):
                self.provider.warmup()
        except Exception as e:
            logger.debug(f"Warmup failed: {e}")
        finally:
            self._warmup_done.set()

    def _cancel_main_loop(self) -> None:
        """Cancel the running listening-loop task and wait for it to stop.

        Called at the start of shutdown() (and by run() on Ctrl+C) so a
        pending loop task is never abandoned — abandoning it leaks the
        in-flight AI stream and raises "Task was destroyed but it is
        pending!" when the loop closes. Idempotent.
        """
        task = self._main_task
        future = self._main_future
        if (task is None and future is None) or self._loop.is_closed():
            return

        if task is not None and not task.done():
            unwound = concurrent.futures.Future()

            def _cancel_and_wait() -> None:
                def _on_done(t: asyncio.Task) -> None:
                    if unwound.cancelled():
                        t.cancel()
                    else:
                        unwound.set_result(None)

                task.add_done_callback(_on_done)
                task.cancel()

            try:
                self._loop.call_soon_threadsafe(_cancel_and_wait)
            except Exception as e:
                logger.warning(f"Could not cancel the main loop task: {e}")
            try:
                unwound.result(timeout=5.0)
            except (asyncio.CancelledError, concurrent.futures.CancelledError, TimeoutError):
                pass
            except Exception as e:
                logger.warning(f"Error stopping the main loop: {e}")

        if future is not None and not future.done():
            try:
                future.cancel()
            except Exception as e:
                logger.warning(f"Could not cancel the main loop future: {e}")

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

        if self.mic.is_available():
            logger.info(f"Microphone: {self.mic.describe()}")
        else:
            logger.warning(
                "Microphone unavailable — commands and text mode "
                "still work."
            )

        if self.ollama_ok:
            logger.info(f"AI engine: {self.provider.describe()}")
            print("[+] AI engine ready")
        else:
            print(
                "[!] Ollama unavailable — local command mode "
                "remains available."
            )
            logger.warning(
                "Ollama not reachable; conversational answers will "
                "be unavailable until it starts."
            )

    def load_models(self) -> None:
        """Load Whisper + Piper voice models once at startup."""
        if self.stt.load():
            print("[+] Speech recognition ready")
        else:
            print(
                "[!] Speech recognition unavailable — text mode "
                "and commands still work."
            )

        if self.tts.load():
            print("[+] Voice engine ready")
        else:
            print(
                "[!] Voice engine unavailable — replies will be "
                "printed to the console instead."
            )

    def initialize_audio(self) -> None:
        """Calibrate the adaptive VAD against ambient noise."""
        calibrate = getattr(self.stt, "vad", None)
        if calibrate is not None and hasattr(calibrate, "calibrate"):
            calibrate.calibrate()
        logger.debug("Audio pipeline ready.")

    def initialize_router(self) -> None:
        """Wire up routing + commands."""
        print("[+] Command router ready")
        logger.debug("Router + command registry ready.")

    # ── Terminal dashboard helpers ────────────────────────────

    def _ui_component(self, name: str, status: str, detail: str) -> None:
        """Update a component row on the terminal dashboard (no-op when
        the dashboard is not active)."""
        if self.ui is not None:
            try:
                self.ui.update_metrics(**{name: f"{status} — {detail}"})
            except Exception:
                pass

    def _ui_state(self, state: str, meta: str = "") -> None:
        """Update the dashboard state indicator (no-op when inactive)."""
        if self.ui is not None:
            try:
                self.ui.update_state(state)
                if meta:
                    self.ui.update_text(meta)
            except Exception:
                pass

    # ── Speech helpers ────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak text via TTS (non-blocking)."""
        if text and text.strip():
            logger.info("[STATE] SPEAKING")
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

    # ── Barge-in / wake word / streaming STT ──────────────────

    def _start_barge_in(self):
        """Start the barge-in monitor when enabled (opt-in: needs a
        quiet room / echo handling). Returns the monitor or None."""
        if not jarvis_config.ENABLE_BARGE_IN:
            return None
        vad = getattr(self.stt, "vad", None)
        if vad is None:
            return None
        try:
            from engine.barge_in import BargeInMonitor

            monitor = BargeInMonitor(
                sample_rate=getattr(vad, "sample_rate", 16000),
                threshold=getattr(vad, "threshold", 500.0),
                input_device=getattr(vad, "input_device", None),
            )
            monitor.start(on_speech=self.tts.stop)
            return monitor
        except Exception as e:
            logger.debug(f"Barge-in monitor unavailable: {e}")
            return None

    async def _await_wake_word(self) -> None:
        """Cooperatively wait for the wake word (when enabled). Never
        blocks the event loop; falls back silently when the optional
        openwakeword package/model is unavailable."""
        if not jarvis_config.ENABLE_WAKE_WORD:
            return
        detector = getattr(self.stt, "wake_word", None)
        if detector is None:
            return
        if detector.available or detector.load():
            phrase = jarvis_config.WAKE_WORD or "the wake word"
            print(f"[Waiting for '{phrase}'...]")
            await asyncio.to_thread(detector.wait_for_wake_word, 60.0)

    async def _listen_voice(self) -> str | None:
        """
        Listen for one utterance.

        With streaming STT enabled, partial transcriptions arrive every
        ~3 seconds while the user is still talking — instant intents
        (e.g. "stop speaking") are acted on immediately, and the final
        transcription drives normal routing.
        """
        streaming = getattr(self.stt, "astream_listen", None)
        if streaming is None or getattr(self.stt, "_streaming", None) is None:
            return await asyncio.to_thread(self.listen)

        handled_early = False
        async for partial, is_final in streaming():
            if is_final:
                return partial
            if not handled_early and partial:
                handled_early = True
                intent, _ = self.router.route(partial)
                if intent == Intent.STOP_SPEECH:
                    self.tts.stop()
        return None

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

    # ── Main listening loop ───────────────────────────────────

    def start_listening_loop(self) -> None:
        """Sync façade: run the async listen → transcribe → route →
        respond loop on the event loop. The future is recorded BEFORE
        blocking so shutdown() can cancel a pending loop cleanly even
        when it races the loop's startup."""
        if self._loop.is_closed():
            asyncio.run(self.astart_listening_loop())
            return
        self._main_future = concurrent.futures.Future()

        def _launch() -> None:
            if self._main_future.done():
                return
            task = asyncio.ensure_future(self.astart_listening_loop())
            self._main_task = task

            def _chain(t: asyncio.Task) -> None:
                if self._main_future.done():
                    t.cancel()
                    return
                if t.cancelled():
                    self._main_future.cancel()
                else:
                    exc = t.exception()
                    if exc is not None:
                        self._main_future.set_exception(exc)
                    else:
                        self._main_future.set_result(t.result())

            task.add_done_callback(_chain)

        try:
            self._loop.call_soon_threadsafe(_launch)
        except RuntimeError:
            asyncio.run(self.astart_listening_loop())
            return
        try:
            self._main_future.result()
        except (asyncio.CancelledError, concurrent.futures.CancelledError):
            return None

    async def astart_listening_loop(self) -> None:
        """The listen → transcribe → route → respond loop (async)."""
        self._main_task = asyncio.current_task()
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

        if self.ui is not None:
            try:
                self.ui.start()
                recorder = getattr(self.stt, "vad", None)
                if recorder is not None:
                    self.ui.update_metrics(vad="active")
            except Exception as e:
                logger.warning(f"Terminal dashboard failed to start: {e}")
                self.ui = None

        consecutive_failures = 0
        MAX_FAILURES = 5

        while self.running:
            turn: dict = {}
            try:
                if self.text_mode:
                    user_input = await asyncio.to_thread(
                        lambda: input("\nYou: ").strip()
                    )
                    if not user_input:
                        continue
                else:
                    barge_in = self._start_barge_in()
                    try:
                        t0 = time.perf_counter()
                        await asyncio.to_thread(self.tts.wait)
                        turn["speak"] = time.perf_counter() - t0
                    finally:
                        if barge_in is not None:
                            barge_in.stop()

                    await self._await_wake_word()

                    logger.info("[STATE] LISTENING")
                    self._ui_state("listening")
                    t0 = time.perf_counter()
                    user_input = await self._listen_voice()
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
                self.process_input(user_input)
                turn["process"] = time.perf_counter() - t0

                if not self.running:
                    break

                self._record_turn(turn)
                await asyncio.sleep(0.1)

            except KeyboardInterrupt:
                logger.info(
                    "[SHUTDOWN] reason=keyboard_interrupt (loop thread)"
                )
                self.running = False
                break

            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                await asyncio.sleep(1)
                continue

    # [FIX M6] Proper shutdown method
    def shutdown(self) -> None:
        """Release resources and exit cleanly."""
        self.running = False
        # Stop the main loop task FIRST — a pending task would keep the
        # AI stream / TTS alive while we tear down and would be
        # destroyed pending on loop close.
        self._cancel_main_loop()
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
        # Stop the dedicated event loop thread.
        if self._loop is not None and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._loop_thread and self._loop_thread.is_alive():
                self._loop_thread.join(timeout=2.0)
            try:
                self._loop.close()
            except Exception:
                pass
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
        except KeyboardInterrupt:
            # Ctrl+C lands on this (main) thread while the loop blocks
            # on the listening coroutine. Cancel the loop task and cut
            # TTS FIRST so the farewell never interleaves with a
            # still-streaming answer, then speak it once.
            logger.info("[SHUTDOWN] reason=keyboard_interrupt")
            print("\n[Shutdown requested]")
            self._cancel_main_loop()
            try:
                self.tts.stop()
            except Exception:
                pass
            try:
                self.speak_blocking(
                    f"Shutting down. "
                    f"Goodbye, {jarvis_config.OWNER}."
                )
            except BaseException:
                # A second Ctrl+C while the farewell is playing must
                # not abort shutdown with a nested traceback.
                pass
        finally:
            self.shutdown()
 
    def process_input(self, text: str) -> str | None:
        """Process user input and return response."""
        validated = validate_input(text)
        if not validated:
            if len(text) > jarvis_config.MAX_INPUT_CHARS:
                self.tts.speak(ERROR_INPUT_TOO_LONG)
                return ERROR_INPUT_TOO_LONG
            return None
        
        # Check for pending confirmation
        if self._pending:
            response = self._handle_confirmation(validated)
            if response is not None:
                return response
            # "other" input: pending dropped — route the fresh input
        
        # Route the input
        intent, data = self.router.route(validated)
        
        if intent == Intent.EXIT:
            self.tts.speak_blocking(f"Goodbye, {jarvis_config.OWNER}.")
            self.running = False
            return None
        
        if intent == Intent.CLEAR_MEMORY:
            self.memory.clear()
            self.tts.speak(ERROR_MEMORY_CLEARED)
            return ERROR_MEMORY_CLEARED
        
        if intent == Intent.STOP_SPEECH:
            self.tts.stop()
            return None
        
        if intent == Intent.FAST_RESPONSE:
            self.tts.speak(data)
            return data
        
        if intent == Intent.MODEL_MODE:
            return self._handle_model_mode(data)
        
        # Check for commands
        cmd_result = self.commands.execute_with_meta(validated)
        if cmd_result:
            if cmd_result.needs_confirmation:
                self._pending = PendingConfirmation(
                    cmd_result, validated
                )
                self.tts.speak(self._pending.prompt)
                return self._pending.prompt
            self.tts.speak(cmd_result.response)
            return cmd_result.response
        
        # Default: AI question
        return self._ask_ai(validated)
 
    def _handle_confirmation(self, text: str) -> str | None:
        """Handle confirmation flow for dangerous commands."""
        decision = "other"
        if _CONFIRM_YES_RE.match(text):
            decision = "yes"
        elif _CONFIRM_NO_RE.match(text):
            decision = "no"
        
        # Extract token if present (for token confirmation)
        token = None
        digits = re.findall(r"\d{4}", text)
        if digits:
            token = digits[0]
        
        pending = self._pending
        original = pending.original_text
        result = pending.take(decision, token)
        self._pending = None
        
        if result:
            try:
                response = result.execute(original)
            except Exception as e:
                logger.warning(f"Confirmed command failed: {e}")
                response = "Sorry, I couldn't complete that command."
            self.tts.speak(response)
            return response
        
        if decision == "other":
            return None
        
        response = "Action cancelled."
        self.tts.speak(response)
        return response
 
    def _handle_model_mode(self, data) -> str:
        """Handle model mode switch/status."""
        action, mode = data
        
        if action == "status":
            model = ollama_config.resolve_model(self.model_mode)
            response = f"I'm using the {self.model_mode} mode with {model}."
            self.tts.speak(response)
            return response
        
        if action == "switch":
            if mode == self.model_mode:
                model = ollama_config.resolve_model(mode)
                response = f"I'm already using {mode} mode with {model}."
                self.tts.speak(response)
                return response
            model = ollama_config.resolve_model(mode)
            if self.provider is not None and hasattr(self.provider, "switch_model"):
                try:
                    self.provider.switch_model(model)
                except Exception as e:
                    logger.warning(f"Model switch failed: {e}")
            self.model_mode = mode
            response = f"Switched to {mode} mode using {model}."
            self.tts.speak(response)
            return response
        
        return ""
 
    def _ask_ai(self, text: str) -> str | None:
        """Ask the AI provider."""
        if not self.provider:
            self.tts.speak(ERROR_OLLAMA_DOWN)
            return ERROR_OLLAMA_DOWN
        
        # Wait for warmup
        self._warmup_done.wait(timeout=30.0)
        
        # Add to memory
        self.memory.add_user_message(text)
        
        try:
            # Check if web search needed
            context = None
            ranked = None
            if self.search and jarvis_config.AI_MODE in ("auto", "web"):
                from brain.classifier import QuestionClassifier
                classifier = QuestionClassifier()
                if classifier.needs_search(text):
                    if not getattr(self.search, "is_configured", lambda: True)():
                        self.tts.speak(ERROR_CANNOT_VERIFY)
                        return ERROR_CANNOT_VERIFY
                    query = build_search_query(text)
                    print("[Searching...]")
                    try:
                        results = self.search.search(query)
                    except Exception as e:
                        logger.warning(f"Web search failed: {e}")
                        self.tts.speak(ERROR_CANNOT_VERIFY)
                        return ERROR_CANNOT_VERIFY
                    ranked = filter_and_rank(results, query)
                    if not ranked:
                        self.tts.speak(ERROR_CANNOT_VERIFY)
                        return ERROR_CANNOT_VERIFY
                    context = format_results_for_llm(ranked)
            
            provider_ok = self.ollama_ok
            try:
                if self.provider and not self.provider.is_available():
                    provider_ok = False
            except Exception as e:
                logger.warning(f"Provider availability check failed: {e}")
            
            # Fallback: answer from the top search snippet when the LLM is down
            if ranked and not provider_ok:
                snippet = " ".join((ranked[0].snippet or ranked[0].title).split())
                if not snippet:
                    self.tts.speak(ERROR_CANNOT_VERIFY)
                    return ERROR_CANNOT_VERIFY
                response = f"Based on my search: {snippet[:280]}"
                print("[Sources]")
                for r in ranked[:3]:
                    if r.url:
                        print(f"  - {r.title or r.source or r.url}")
                        print(f"    {r.url}")
                self.memory.add_assistant_message(response)
                self.tts.speak(response)
                return response
            
            # Provider offline and nothing to fall back on.
            if not provider_ok:
                self.tts.speak(ERROR_OLLAMA_DOWN)
                return ERROR_OLLAMA_DOWN
            
            # Get response
            response = self.provider.ask(
                text,
                memory=self.memory,
                context=context,
            )
            
            if context:
                print("[Sources]")
                for r in ranked[:3]:
                    if r.url:
                        print(f"  - {r.title or r.source or r.url}")
                        print(f"    {r.url}")
            
            if response:
                self.memory.add_assistant_message(response)
                self.tts.speak(response)
                return response
            else:
                self.tts.speak(ERROR_AI_FAILED)
                return ERROR_AI_FAILED
                
        except OllamaTimeoutError:
            self.tts.speak(ERROR_OLLAMA_DOWN)
            return ERROR_OLLAMA_DOWN
        except Exception as e:
            logger.error(f"AI request failed: {e}")
            self.tts.speak(ERROR_AI_FAILED)
            return ERROR_AI_FAILED
 
 
def run_assistant(
    text_mode: bool = False,
    debug: bool = False,
    benchmark: bool = False,
) -> int:
    """Entry point used by the `jarvis` CLI."""
    jarvis = JARVIS(text_mode=text_mode, debug=debug, benchmark=benchmark)
    try:
        jarvis.run()
    except KeyboardInterrupt:
        pass
    finally:
        jarvis.shutdown()
    return 0


def main():
    """Entry point."""
    import argparse
    parser = argparse.ArgumentParser(description="JARVIS AI Assistant")
    parser.add_argument("--text", action="store_true", help="Text mode (no microphone)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--benchmark", action="store_true", help="Print timing stats")
    args = parser.parse_args()
    
    jarvis = JARVIS(
        text_mode=args.text,
        debug=args.debug,
        benchmark=args.benchmark,
    )
    
    try:
        jarvis.run()
    except KeyboardInterrupt:
        pass
    finally:
        jarvis.shutdown()
 
 
if __name__ == "__main__":
    main()