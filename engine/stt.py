"""
engine/stt.py — Speech-to-Text engine

Captures audio with sounddevice and sends RAW PCM directly
to Google's speech API using requests.

Why not use speech_recognition.recognize_google()?
  It converts WAV -> FLAC by running a bundled flac.exe
  subprocess. On Windows ARM64 / restricted systems that
  binary fails with [WinError 50] The request is not supported.

This implementation sends audio/l16 (raw 16-bit PCM) which
Google accepts natively. No FLAC. No subprocess. No PyAudio.
"""

import json
import time
import numpy as np
import sounddevice as sd
import requests
from typing import Optional
from utils.logger import get_logger

logger = get_logger("stt")

# ── Audio capture settings ────────────────────────────────────
SAMPLE_RATE = 16000        # Google speech API expects 16 kHz
CHANNELS = 1               # Mono
DTYPE = "int16"            # 16-bit PCM
SILENCE_THRESHOLD = 500    # RMS below this = silence
SILENCE_DURATION = 0.7     # Seconds of silence ends the phrase
CHUNK_DURATION = 0.05      # Audio chunk length in seconds
MIN_SPEECH_CHUNKS = 3      # Ignore very short blips

# ── Google speech endpoint (same one SpeechRecognition uses) ──
GOOGLE_URL = (
    "http://www.google.com/speech-api/v2/recognize"
    "?client=chromium&lang={lang}&key={key}"
)

# ── Load config safely ────────────────────────────────────────
try:
    from config import stt_config
    LANGUAGE = stt_config.LANGUAGE
    TIMEOUT = stt_config.TIMEOUT
    PHRASE_LIMIT = stt_config.PHRASE_LIMIT
    SILENCE_DURATION = stt_config.SILENCE_DURATION
    CHUNK_DURATION = stt_config.CHUNK_DURATION
    GOOGLE_KEY = stt_config.GOOGLE_KEY
except Exception:
    LANGUAGE = "en-US"
    TIMEOUT = 5
    PHRASE_LIMIT = 10
    SILENCE_DURATION = 0.7
    CHUNK_DURATION = 0.05
    GOOGLE_KEY = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"


class STTEngine:
    """
    Speech-to-text using sounddevice capture and a direct
    HTTP request to Google's speech API.
    """

    def __init__(self, mic_manager=None):
        # mic_manager kept for compatibility, not required
        self.sample_rate = SAMPLE_RATE
        self.channels = CHANNELS
        self.dtype = DTYPE
        self.language = LANGUAGE
        self._test_microphone()

    def _test_microphone(self) -> None:
        """Log which microphone will be used."""
        try:
            default_input = sd.default.device[0]
            if default_input is None:
                logger.warning("No default input device found.")
                return
            info = sd.query_devices(default_input)
            logger.info(
                f"Microphone: {info.get('name', 'Unknown')}"
            )
        except Exception as e:
            logger.error(f"Microphone check failed: {e}")

    # ── Public API ────────────────────────────────────────────

    def listen(self) -> Optional[str]:
        """
        Record one phrase from the microphone and transcribe it.

        Returns:
            str  : recognized text (lowercase)
            None : nothing heard, or recognition failed
        """
        t0 = time.perf_counter()
        audio = self._record_phrase()
        if audio is None:
            return None

        t1 = time.perf_counter()
        pcm_bytes = audio.tobytes()
        text = self._recognize(pcm_bytes)
        t2 = time.perf_counter()

        logger.info(
            f"[timing] record {(t1 - t0) * 1000:.0f}ms | "
            f"recognize {(t2 - t1) * 1000:.0f}ms"
        )

        if text:
            logger.info(f"Recognized: '{text}'")
            print(f"[You said: {text}]")
        else:
            logger.debug("No transcription returned.")

        return text

    # ── Audio capture ─────────────────────────────────────────

    def _record_phrase(self) -> Optional[np.ndarray]:
        """
        Record until the user stops speaking.

        Returns numpy int16 array, or None on timeout/error.
        """
        try:
            print("\n[Listening... speak now]")

            chunk_duration = CHUNK_DURATION
            chunk_size = int(self.sample_rate * chunk_duration)
            max_silence = int(SILENCE_DURATION / chunk_duration)
            max_total = int(PHRASE_LIMIT / chunk_duration)
            max_wait = int(TIMEOUT / chunk_duration)

            chunks = []
            silence_count = 0
            wait_count = 0
            speaking = False

            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                blocksize=chunk_size
            ) as stream:

                while True:
                    raw, _ = stream.read(chunk_size)

                    # Convert cffi buffer -> numpy (Py 3.14 safe)
                    chunk = np.frombuffer(
                        bytes(raw), dtype=np.int16
                    )

                    rms = self._rms(chunk)

                    if not speaking:
                        wait_count += 1
                        if wait_count > max_wait:
                            return None      # nobody spoke
                        if rms > SILENCE_THRESHOLD:
                            speaking = True
                            chunks.append(chunk)
                            print("[Speech detected...]")
                    else:
                        chunks.append(chunk)
                        if rms < SILENCE_THRESHOLD:
                            silence_count += 1
                        else:
                            silence_count = 0

                        if silence_count >= max_silence:
                            break
                        if len(chunks) >= max_total:
                            break

            if len(chunks) < MIN_SPEECH_CHUNKS:
                logger.debug("Speech too short, ignoring.")
                return None

            print("[Processing...]")
            return np.concatenate(chunks, axis=0)

        except sd.PortAudioError as e:
            logger.error(f"PortAudio error: {e}")
            return None
        except Exception as e:
            logger.error(f"Recording error: {e}")
            return None

    def _rms(self, chunk: np.ndarray) -> float:
        """Volume level of an audio chunk."""
        if chunk.size == 0:
            return 0.0
        return float(
            np.sqrt(np.mean(chunk.astype(np.float32) ** 2))
        )

    # ── Recognition ───────────────────────────────────────────

    def _recognize(self, pcm_bytes: bytes) -> Optional[str]:
        """
        Send raw 16-bit PCM to Google and parse the response.

        Google returns several newline-separated JSON objects.
        Only one of them contains the transcript.
        """
        url = GOOGLE_URL.format(
            lang=self.language, key=GOOGLE_KEY
        )
        headers = {
            "Content-Type": f"audio/l16; rate={self.sample_rate}"
        }

        try:
            resp = requests.post(
                url,
                data=pcm_bytes,
                headers=headers,
                timeout=15
            )

            if resp.status_code != 200:
                logger.error(
                    f"Google STT HTTP {resp.status_code}"
                )
                return None

            # Parse newline-delimited JSON
            for line in resp.text.strip().split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                results = data.get("result", [])
                if not results:
                    continue

                alternatives = results[0].get("alternative", [])
                if not alternatives:
                    continue

                transcript = alternatives[0].get("transcript", "")
                if transcript.strip():
                    return transcript.strip().lower()

            logger.debug("Google returned no transcript.")
            return None

        except requests.Timeout:
            logger.error("Google STT request timed out.")
            return None
        except requests.ConnectionError:
            logger.error(
                "No internet connection for speech recognition."
            )
            return None
        except Exception as e:
            logger.error(f"Recognition error: {e}")
            return None