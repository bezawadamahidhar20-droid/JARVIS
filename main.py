"""JARVIS — hands-free local voice assistant main loop.

Pipeline: VAD capture -> Faster-Whisper -> command router / Qwen3 -> Piper TTS

Design notes
------------
* Exit detection lives **only** here, in ``is_exit_phrase()``.  The command
  router (``commands/router.py``) deliberately has no exit logic, so sentences
  that contain the words "exit" or "quit" mid-phrase ("what is an exit code?",
  "how do I quit vim?") are safely forwarded to the LLM instead of shutting
  JARVIS down.

* The post-TTS microphone cooldown (``mic_cooldown()``) pauses after
  ``tts.speak()`` returns so the speaker tail / room reverb has time to decay
  before the VAD re-arms.  Without this hold-off the VAD can pick up JARVIS's
  own voice and feed it back into Whisper as phantom speech.
"""

import re
import time

import sounddevice as sd

import config
from ai.ollama import (
    ModelMissingError,
    OllamaClient,
    OllamaResponseError,
    OllamaUnavailableError,
)
from audio.microphone import SpeechRecorder
from commands.router import CommandRouter
from speech.tts import TTSEngine
from speech.whisper import WhisperEngine
from utils import dataset, logger

# ── Constants ─────────────────────────────────────────────────────────────────

# Closed allow-list of whole-utterance exit phrases.
# Words like "exit" or "quit" must match the ENTIRE normalised utterance to
# trigger shutdown — not just appear somewhere inside a longer sentence.
EXIT_PHRASES: frozenset[str] = frozenset({
    "exit",
    "quit",
    "goodbye",
    "stop",
    "stop jarvis",
    "shut down jarvis",
})

BANNER: str = (
    "============================\n"
    "       JARVIS ONLINE\n"
    "============================\n"
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _words(text: str) -> str:
    """Normalise *text* to lowercase letters and spaces only."""
    return " ".join(re.sub(r"[^a-z ]", "", text.strip().lower()).split())


def is_exit_phrase(text: str) -> bool:
    """Return ``True`` only when the *entire* utterance is an exit phrase.

    This guards against false shutdowns from sentences that merely contain
    words like "exit" or "quit":
        "what is an exit code?"   → False  (forwarded to LLM)
        "how do I quit vim?"      → False  (forwarded to LLM)
        "goodbye"                 → True   (shuts down)
        "exit"                    → True   (shuts down)
    """
    return _words(text) in EXIT_PHRASES


def is_meaningful(text: str) -> bool:
    """Reject noise transcriptions without blocking short commands.

    A transcription must have at least 2 characters total and at least 2
    alphabetic characters to be considered real speech.
    """
    t = text.strip()
    return len(t) >= 2 and sum(c.isalpha() for c in t) >= 2


def mic_cooldown(seconds: float = config.POST_TTS_COOLDOWN_S) -> None:
    """Pause after TTS playback before the mic re-arms.

    After ``sd.wait()`` returns in ``TTSEngine.speak()`` the speaker cone is
    still moving and the room reverb tail has not decayed.  This hold-off
    prevents the VAD from picking up JARVIS's own voice on the next loop
    iteration and feeding it into Whisper as a phantom utterance.

    Set ``POST_TTS_COOLDOWN_S = 0`` in ``config.py`` for headphone-only use
    or when echo cancellation is handled externally.
    """
    if seconds > 0:
        time.sleep(seconds)


# ── Main loop ─────────────────────────────────────────────────────────────────

def main() -> None:
    print(BANNER)

    # ── Microphone availability ───────────────────────────────────────────────
    try:
        sd.query_devices(kind="input")
    except Exception as exc:
        logger.error(f"Microphone unavailable: {exc}")
        return

    # ── Whisper ───────────────────────────────────────────────────────────────
    try:
        whisper = WhisperEngine(
            model_size=config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
            language=config.WHISPER_LANGUAGE,
            beam_size=config.WHISPER_BEAM_SIZE,
        )
    except Exception as exc:
        logger.error(f"Whisper failed to load: {exc}")
        return

    # ── Ollama (non-fatal if down — local commands still work) ────────────────
    ollama = OllamaClient()
    try:
        ollama.check_available()
    except (OllamaUnavailableError, ModelMissingError, OllamaResponseError) as exc:
        logger.error(str(exc))
        logger.warning("Conversational questions will not work until Ollama is fixed.")

    # ── Command router ────────────────────────────────────────────────────────
    router = CommandRouter()
    logger.ok("Command router ready")

    # ── Conversation dataset (fine-tuning) ────────────────────────────────────
    dataset_logger = dataset.ConversationDataset(
        path=config.DATASET_PATH,
        system_prompt=config.SYSTEM_PROMPT,
    )

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts = TTSEngine()
    try:
        tts.initialize()
    except Exception as exc:
        logger.error(f"TTS unavailable: {exc}")

    # ── Microphone recorder ───────────────────────────────────────────────────
    recorder = SpeechRecorder(
        sample_rate=config.SAMPLE_RATE,
        channels=config.CHANNELS,
        input_device=config.INPUT_DEVICE,
        frame_ms=config.FRAME_MS,
        min_speech_ms=config.MIN_SPEECH_MS,
        silence_ms=config.SILENCE_MS,
        max_record_ms=config.MAX_RECORD_MS,
        calibration_ms=config.CALIBRATION_MS,
        flush_ms=config.STREAM_FLUSH_MS,
        initial_threshold=config.INITIAL_RMS_THRESHOLD,
        multiplier=config.NOISE_MULTIPLIER,
        alpha=config.NOISE_EMA_ALPHA,
    )

    print()
    logger.status("JARVIS is listening...\n")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        turn_start: float = logger.tick()
        logger.status("Listening...")

        # 1. Capture utterance via VAD-gated streaming mic.
        try:
            audio = recorder.capture_speech()
        except sd.PortAudioError as exc:
            logger.error(f"Microphone error: {exc}")
            logger.error("Shutting down — restart JARVIS after fixing the microphone.")
            return
        logger.report("CAPTURE", turn_start)
        logger.info("[VOICE] Speech detected")

        if len(audio) == 0:
            logger.warning("No speech detected\n")
            continue

        # 2. Transcribe with Faster-Whisper.
        w_start: float = logger.tick()
        try:
            text: str = whisper.transcribe(audio)
        except Exception as exc:
            logger.error(f"Whisper failed: {exc}\n")
            continue
        logger.report("WHISPER", w_start)

        print(f"[USER] {text}")

        # 3. Noise filter.
        if not is_meaningful(text):
            logger.warning("No speech detected\n")
            continue

        # 4. Exit check — whole-utterance match only.
        #    Words like "exit" or "quit" inside a normal sentence never match
        #    because is_exit_phrase() compares the FULL normalised string.
        if is_exit_phrase(text):
            tts.speak("Goodbye.")
            print("[*] JARVIS shutting down.")
            break

        # 5. Deterministic command routing.
        r_start: float = logger.tick()
        command, _ = router.route(text)
        logger.report("ROUTER", r_start)

        # 6. Execute command or ask Qwen3.
        response: str
        if command is not None:
            logger.info(f"[COMMAND] {command}")
            try:
                response = router.execute(command)
            except Exception as exc:
                logger.error(f"Command execution failed: {exc}")
                response = "Sorry, I could not complete that command."
        else:
            try:
                o_start: float = logger.tick()
                response = ollama.generate(text)
                logger.report("OLLAMA", o_start)
            except OllamaUnavailableError as exc:
                logger.error(str(exc))
                response = "Ollama is not running, so I cannot answer that right now."
            except ModelMissingError as exc:
                logger.error(str(exc))
                response = "The Qwen model is not installed on this Ollama."
            except OllamaResponseError as exc:
                logger.error(str(exc))
                response = "Ollama returned an error while answering."
            except Exception as exc:
                logger.error(f"Ollama failed: {exc}")
                response = "Sorry, something went wrong on my end."

        # 7. Speak and record the exchange for multi-turn context.
        print(f"[JARVIS] {response}")
        tts.speak(response)
        ollama.add_turn(text, response)
        # Collect real data for a future fine-tune.
        dataset_logger.record(text, response)
        logger.report("TOTAL", turn_start)
        print()

        # 8. Post-TTS hold-off: let the speaker tail decay before re-arming.
        mic_cooldown()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.status("JARVIS stopped.")
