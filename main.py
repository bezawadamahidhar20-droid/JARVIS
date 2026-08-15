"""JARVIS — hands-free local voice assistant main loop.

Pipeline: VAD capture -> Faster-Whisper -> command router / Qwen3 -> Piper TTS
"""

import re

import sounddevice as sd

import config
from ai.ollama import (
    OllamaClient,
    ModelMissingError,
    OllamaResponseError,
    OllamaUnavailableError,
)
from audio.microphone import SpeechRecorder
from commands.router import CommandRouter
from speech.tts import TTSEngine
from speech.whisper import WhisperEngine
from utils import logger

EXIT_PHRASES = ("exit", "quit", "goodbye", "stop", "stop jarvis", "shut down jarvis")

BANNER = (
    "============================\n"
    "       JARVIS ONLINE\n"
    "============================\n"
)


def _words(text):
    return " ".join(re.sub(r"[^a-z ]", "", text.strip().lower()).split())


def is_exit_phrase(text):
    """True only when the whole utterance is an exit phrase.

    Guards against false exits from sentences that merely contain
    words like "exit" or "quit" ("what is an exit code?").
    """
    return _words(text) in EXIT_PHRASES


def is_meaningful(text):
    """Reject noise transcriptions without blocking short commands."""
    t = text.strip()
    return len(t) >= 2 and sum(c.isalpha() for c in t) >= 2


def main():
    print(BANNER)

    try:
        sd.query_devices(kind="input")
    except Exception as exc:
        logger.error(f"Microphone unavailable: {exc}")
        return

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

    ollama = OllamaClient()
    try:
        ollama.check_available()
    except (OllamaUnavailableError, ModelMissingError, OllamaResponseError) as exc:
        logger.error(str(exc))
        logger.warning("Conversational questions will not work until Ollama is fixed.")

    router = CommandRouter()
    logger.ok("Command router ready")

    tts = TTSEngine()
    try:
        tts.initialize()
    except Exception as exc:
        logger.error(f"TTS unavailable: {exc}")

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

    while True:
        turn_start = logger.tick()
        logger.status("Listening...")

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

        w_start = logger.tick()
        try:
            text = whisper.transcribe(audio)
        except Exception as exc:
            logger.error(f"Whisper failed: {exc}\n")
            continue
        logger.report("WHISPER", w_start)

        print(f"[USER] {text}")

        if not is_meaningful(text):
            logger.warning("No speech detected\n")
            continue

        if is_exit_phrase(text):
            tts.speak("Goodbye.")
            print("[*] JARVIS shutting down.")
            break

        r_start = logger.tick()
        command, _ = router.route(text)
        logger.report("ROUTER", r_start)

        if command is not None:
            logger.info(f"[COMMAND] {command}")
            try:
                response = router.execute(command)
            except Exception as exc:
                logger.error(f"Command execution failed: {exc}")
                response = "Sorry, I could not complete that command."
        else:
            try:
                o_start = logger.tick()
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

        print(f"[JARVIS] {response}")
        tts.speak(response)
        logger.report("TOTAL", turn_start)
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        logger.status("JARVIS stopped.")