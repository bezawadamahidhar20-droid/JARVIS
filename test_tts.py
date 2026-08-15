"""Standalone test for the local TTS engine.

Speaks:  Hello. I am JARVIS. Text to speech is working.
"""

import config
from speech.tts import TTSEngine
from utils import logger


def main():
    engine = TTSEngine(voice_path=config.TTS_VOICE_PATH)
    engine.initialize()

    text = "Hello. I am JARVIS. Text to speech is working."
    print(f"[>] Speaking: {text}\n")
    engine.speak(text)

    logger.ok("TTS test done — if you heard JARVIS, TTS works.")


if __name__ == "__main__":
    main()