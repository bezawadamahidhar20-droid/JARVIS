"""JARVIS — voice assistant main loop.

Pipeline:  microphone → speech-to-text → intent router → local command
           OR Qwen3 (Ollama) → text-to-speech

Modes
-----
* Voice (default): listens through the microphone.
* ``--text``      : type your input instead — perfect for testing, CI or
                    machines without a microphone.
"""

import argparse
import sys

import config
from brain.memory import ConversationMemory
from brain.ollama_client import (
    ModelMissingError,
    OllamaClient,
    OllamaResponseError,
    OllamaUnavailableError,
)
from brain.router import AI_QUESTION, CLEAR_MEMORY, COMMAND, EXIT, IntentRouter
from commands.registry import CommandRegistry
from engine.microphone import MicrophoneManager
from engine.stt import STTEngine
from engine.tts import TTSEngine
from utils import logger

BANNER: str = (
    "=============================================\n"
    "              JARVIS ONLINE\n"
    "============================================="
)


class JARVIS:
    """Wires every component together and runs the conversation loop."""

    def __init__(self, text_mode: bool = False) -> None:
        self.text_mode = text_mode

        # Brain
        self.memory = ConversationMemory()
        self.ollama = OllamaClient()
        self.router = IntentRouter()

        # Commands
        self.commands = CommandRegistry()

        # Audio
        self.tts = TTSEngine(rate=config.TTS_RATE)
        self.mic = MicrophoneManager()
        self.stt = STTEngine(
            self.mic,
            language=config.STT_LANGUAGE,
            timeout=config.STT_TIMEOUT,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def speak(self, text: str) -> None:
        """Speak and print *text* (TTS degrades gracefully to print-only)."""
        self.tts.speak(text)

    def _get_input(self) -> str | None:
        """Return one utterance from voice or text mode (None = nothing heard)."""
        if self.text_mode:
            try:
                return input("You: ").strip()
            except EOFError:
                return None
        return self.stt.listen()

    def _answer_with_ai(self, text: str) -> str:
        """Send *text* to Qwen3 with full memory context; never raises."""
        try:
            self.memory.add_user_message(text)
            response = self.ollama.ask(text, self.memory)
            self.memory.add_assistant_message(response)
            return response
        except (OllamaUnavailableError, ModelMissingError, OllamaResponseError) as exc:
            logger.error(str(exc))
            return (
                "My AI brain is offline right now — I can't reach the local "
                "Ollama model. Please start Ollama and I'll be back to normal."
            )
        except Exception as exc:
            logger.error(f"Unexpected AI failure: {exc}")
            return "Sorry, something went wrong on my end."

    def _greeting(self) -> str:
        return (
            f"Good day, {config.JARVIS_OWNER}. I am JARVIS, at your service. "
            "How can I help you today?"
        )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self) -> None:
        print(BANNER)
        logger.ok(f"Model: {config.OLLAMA_MODEL}")

        if not self.ollama.is_available():
            logger.warning(
                "Ollama is not running — general questions won't work until "
                "you start it (commands like time and open still will)."
            )

        if self.text_mode:
            logger.status("Text mode — type your question and press Enter.")
        else:
            if not self.mic.is_available():
                logger.error(
                    "No microphone found. Run with --text to type instead."
                )
                return
            self.stt.calibrate()
            logger.status("JARVIS is listening...")

        # Startup greeting.
        self.speak(self._greeting())
        print()

        while True:
            try:
                text = self._get_input()
            except KeyboardInterrupt:
                self.speak("Goodbye.")
                break

            if text is None:
                # Voice: silence/timeout → keep listening.
                # Text: EOF (stdin closed) → end the session.
                if self.text_mode:
                    break
                continue
            if not text.strip():
                continue

            text = text.strip()
            print(f"[You] {text}")

            intent, text = self.router.route(text)

            if intent == EXIT:
                self.speak("Goodbye. See you soon.")
                break

            if intent == CLEAR_MEMORY:
                self.memory.clear()
                self.speak("I've cleared our conversation memory.")
                continue

            if intent == COMMAND:
                try:
                    response = self.commands.execute(text)
                except Exception as exc:
                    logger.error(f"Command execution failed: {exc}")
                    response = "Sorry, I couldn't complete that command."
            else:  # AI_QUESTION — the default for anything unmatched
                response = self._answer_with_ai(text)

            self.speak(response)
            print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JARVIS — local AI voice assistant")
    parser.add_argument(
        "--text",
        action="store_true",
        help="Run in text mode (type input instead of using the microphone).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    assistant = JARVIS(text_mode=args.text)
    try:
        assistant.run()
    except KeyboardInterrupt:
        print()
        logger.status("JARVIS stopped.")


if __name__ == "__main__":
    sys.exit(main())