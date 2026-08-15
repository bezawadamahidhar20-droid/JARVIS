"""
main.py — JARVIS AI Assistant Entry Point

Flow:
  Microphone → STT → IntentRouter → [FastResponse | Command | Qwen3] → TTS

Speed:
  * Model warm-up runs in a background thread at startup, so the first
    question has no cold-start delay.
  * Voice replies STREAM: each finished sentence is spoken immediately
    while the rest of the answer is still being generated.
  * Greetings get an instant canned reply — no AI round-trip at all.
"""

import sys
import time
import threading

# ── Config ────────────────────────────────────────────────────
from config import jarvis_config, tts_config

# ── Engine ────────────────────────────────────────────────────
from engine.microphone import MicrophoneManager
from engine.stt import STTEngine
from engine.tts import TTSEngine

# ── Brain ─────────────────────────────────────────────────────
from brain.memory import ConversationMemory
from brain.ollama_client import OllamaClient
from brain.router import IntentRouter, Intent

# ── Commands ──────────────────────────────────────────────────
from commands.registry import CommandRegistry

# ── Utils ─────────────────────────────────────────────────────
from utils.logger import get_logger

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


class JARVIS:
    """
    Main JARVIS orchestrator.
    Wires all components together into a single pipeline.
    """

    def __init__(self, text_mode: bool = False):
        mode = "TEXT MODE" if text_mode else "VOICE MODE"
        print("\n=============================================")
        print("              JARVIS ONLINE")
        print(f"              {mode}")
        print("=============================================")

        self.text_mode = text_mode

        # Audio pipeline
        self.mic = MicrophoneManager()
        self.stt = STTEngine(self.mic)
        self.tts = TTSEngine(rate=tts_config.RATE)

        # AI brain
        self.memory = ConversationMemory()
        self.ollama = OllamaClient()

        # Routing and commands
        self.router = IntentRouter()
        self.commands = CommandRegistry()

        self.running = False
        self._start_warmup()
        logger.info("JARVIS initialized successfully.")

    # ── Startup ───────────────────────────────────────────────

    def _start_warmup(self) -> None:
        """
        Pre-load the model in a background thread so the first
        real question has no cold-start delay.
        """
        if not jarvis_config.ENABLE_WARMUP:
            logger.info("Warm-up disabled (ENABLE_WARMUP=false).")
            return
        if not self.ollama.is_available():
            logger.warning("Ollama not reachable; skipping warm-up.")
            return
        threading.Thread(
            target=self.ollama.warmup,
            name="jarvis-warmup",
            daemon=True,
        ).start()

    # ── Speech helpers ────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak text via TTS (non-blocking)."""
        if text and text.strip():
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
        Send question to Qwen3 via Ollama.
        Updates memory before and after for context tracking.
        This is the core fix — ALL general questions reach here.
        """
        if not self.ollama.is_available():
            logger.error("Ollama not available.")
            self.speak(ERROR_OLLAMA_DOWN)
            return

        # Add to memory BEFORE asking
        self.memory.add_user_message(user_input)

        print("[Thinking...]")
        t0 = time.perf_counter()

        if self.text_mode:
            response = self.ollama.ask(user_input, self.memory)
        else:
            # Stream: each finished sentence is spoken immediately
            # while the rest of the answer is still generating.
            response = self.ollama.ask_stream(
                user_input,
                self.memory,
                on_sentence=self.tts.speak,
            )

        logger.info(
            f"[timing] interaction {(time.perf_counter() - t0):.1f}s"
        )

        if response:
            # Add response to memory for future context
            self.memory.add_assistant_message(response)
            if self.text_mode:
                self.speak(response)
        else:
            # Remove failed user message from memory
            if self.memory._messages:
                self.memory._messages.pop()
            self.speak(ERROR_AI_FAILED)

    def handle_command(self, user_input: str) -> None:
        """Execute a system command."""
        result = self.commands.execute(user_input)
        if result:
            self.speak(result)

    def process_input(self, user_input: str) -> bool:
        """
        Process one user input through the full pipeline.

        Returns:
            False = exit JARVIS
            True  = continue running
        """
        if not user_input or not user_input.strip():
            return True

        print(f"\n[You] {user_input}")

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
            self.handle_command(cleaned)

        else:
            # AI_QUESTION or UNKNOWN — send to Qwen3
            # This is the KEY fix: default to AI
            self.handle_ai_question(user_input)

        return True

    # ── Main loop ─────────────────────────────────────────────

    def run(self) -> None:
        """Main loop. Handles both voice and --text modes."""
        self.running = True
        self.greet()

        if self.text_mode:
            print("\n=============================================")
            print("  JARVIS — TEXT MODE")
            print("  Type your message and press Enter.")
            print("  Type 'exit' to quit.")
            print("=============================================\n")
        else:
            print("\n=============================================")
            print(f"  {jarvis_config.NAME} is ready.")
            print("  Speak a command or question.")
            print("  Say 'goodbye' to exit.")
            print("=============================================\n")

        consecutive_failures = 0
        MAX_FAILURES = 5

        while self.running:
            try:
                if self.text_mode:
                    user_input = input("\nYou: ").strip()
                    if not user_input:
                        continue
                else:
                    # Never listen while JARVIS is still speaking,
                    # or the mic will pick up JARVIS's own voice (echo).
                    self.tts.wait()
                    user_input = self.listen()
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

                should_continue = self.process_input(user_input)
                if not should_continue:
                    self.running = False
                    break

                time.sleep(0.1)

            except KeyboardInterrupt:
                self.speak_blocking(
                    f"Shutting down. "
                    f"Goodbye, {jarvis_config.OWNER}."
                )
                self.running = False
                break

            except Exception as e:
                logger.error(
                    f"Unexpected error in main loop: {e}",
                    exc_info=True
                )
                time.sleep(1)
                continue

        logger.info("JARVIS stopped.")


def run_text_mode() -> None:
    """
    Text mode for testing without microphone.
    Run with: python main.py --text
    """
    JARVIS(text_mode=True).run()


if __name__ == "__main__":
    if "--text" in sys.argv or "--text-mode" in sys.argv:
        run_text_mode()
    else:
        jarvis = JARVIS()
        jarvis.run()