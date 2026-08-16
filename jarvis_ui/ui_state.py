"""Shared state + threading bridge between the JARVIS backend and the UI.

Threading model
---------------
* Main thread            : the Qt event loop (UI rendering, 60fps).
* Backend thread         : the JARVIS audio pipeline (mic -> STT -> Ollama -> TTS).
* Stats thread           : CPU / RAM sampling every 2 seconds.
* Audio monitor thread   : microphone level monitoring for the waveform.

Communication is done with two :class:`queue.Queue` instances:

* ``ui_queue``   — backend -> UI.  The UI drains it every frame and applies
                   the events to :class:`JARVISState`.
* ``input_queue`` — UI -> backend.  Text commands typed by the user.

The backend thread never touches Qt widgets; it only mutates the (thread-safe)
state object and puts plain dicts on ``ui_queue``.  That keeps the UI fluid
even while Ollama is busy generating a reply.
"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any, Optional

import numpy as np

# ── Status constants ──────────────────────────────────────────────────────────

IDLE = "idle"
LISTENING = "listening"
THINKING = "thinking"
SPEAKING = "speaking"
ERROR = "error"
SHUTDOWN = "shutdown"

STATUS_LABELS = {
    IDLE: "IDLE",
    LISTENING: "LISTENING...",
    THINKING: "THINKING...",
    SPEAKING: "SPEAKING...",
    ERROR: "ERROR",
    SHUTDOWN: "SHUTDOWN",
}


# ── Event helpers ─────────────────────────────────────────────────────────────

def status_event(value: str) -> dict[str, Any]:
    return {"event": "status_change", "value": value}


def module_event(name: str, active: bool) -> dict[str, Any]:
    return {"event": "module", "name": name, "active": active}


def stats_event(cpu: float, ram: float) -> dict[str, Any]:
    return {"event": "stats", "cpu": cpu, "ram": ram}


# ── JARVISState ───────────────────────────────────────────────────────────────

class JARVISState:
    """Thread-safe snapshot of everything the UI renders."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

        self.status = IDLE
        self.current_input = ""
        self.current_response = ""
        self.conversation_history: list[dict[str, str]] = []

        self.ollama_connected = False
        self.mic_available = False
        self.tts_available = True

        self.cpu_usage = 0.0
        self.ram_usage = 0.0
        self.response_time = 0.0
        self.memory_turns = 0

        self.audio_level = 0.0
        self.audio_level_buf: list[float] = []
        self.max_audio_buf = 96

        self.current_intent = "AI_QUESTION"
        self.confidence = 0.0
        self.route = "Qwen3 Brain"
        self.command_history: list[str] = []
        self.thinking_progress = 0.0
        self.tokens_per_sec = 0.0

        self.active_module: Optional[str] = None
        self.module_activity: dict[str, float] = {}

        self.mic_enabled = True
        self.glitch_active = False
        self.glitch_seed = 0.0

    # ── Locked accessors ────────────────────────────────────────────────────

    def set_status(self, value: str) -> None:
        with self._lock:
            self.status = value
            if value != SPEAKING:
                self.audio_level = 0.0

    def get_status(self) -> str:
        with self._lock:
            return self.status

    def set_audio_level(self, level: float) -> None:
        with self._lock:
            self.audio_level = max(0.0, min(1.0, level))
            self.audio_level_buf.append(self.audio_level)
            if len(self.audio_level_buf) > self.max_audio_buf:
                self.audio_level_buf.pop(0)

    def get_audio_level(self) -> float:
        with self._lock:
            return self.audio_level

    def get_audio_buffer(self) -> list[float]:
        with self._lock:
            return list(self.audio_level_buf)

    def push_conversation(self, role: str, text: str) -> None:
        with self._lock:
            self.conversation_history.append({"role": role, "text": text})
            if len(self.conversation_history) > 40:
                self.conversation_history = self.conversation_history[-40:]

    def get_conversation(self) -> list[dict[str, str]]:
        with self._lock:
            return list(self.conversation_history)

    def clear_conversation(self) -> None:
        with self._lock:
            self.conversation_history.clear()
            self.command_history.clear()
            self.memory_turns = 0

    def add_command_history(self, text: str) -> None:
        with self._lock:
            self.command_history.append(text)
            if len(self.command_history) > 30:
                self.command_history = self.command_history[-30:]

    def set_module(self, name: str, active: bool) -> None:
        with self._lock:
            if active:
                self.active_module = name
                self.module_activity[name] = 1.0
            else:
                self.module_activity[name] = 0.0
                if self.active_module == name:
                    self.active_module = None

    def get_module_activity(self) -> dict[str, float]:
        with self._lock:
            return dict(self.module_activity)

    def get_active_module(self) -> Optional[str]:
        with self._lock:
            return self.active_module

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "status": self.status,
                "current_input": self.current_input,
                "current_response": self.current_response,
                "ollama_connected": self.ollama_connected,
                "mic_available": self.mic_available,
                "tts_available": self.tts_available,
                "cpu_usage": self.cpu_usage,
                "ram_usage": self.ram_usage,
                "response_time": self.response_time,
                "memory_turns": self.memory_turns,
                "audio_level": self.audio_level,
                "current_intent": self.current_intent,
                "confidence": self.confidence,
                "route": self.route,
                "thinking_progress": self.thinking_progress,
                "tokens_per_sec": self.tokens_per_sec,
                "mic_enabled": self.mic_enabled,
            }


# ── Backend controller ────────────────────────────────────────────────────────

class JARVISController:
    """Runs the real JARVIS pipeline in a background thread.

    If the backend modules cannot be imported (for example when the UI is run
    standalone) a lightweight *simulation* loop keeps the UI alive so the
    interface can be previewed without a microphone or Ollama.
    """

    def __init__(
        self,
        state: JARVISState,
        ui_queue: queue.Queue,
        input_queue: queue.Queue,
    ) -> None:
        self.state = state
        self.ui_queue = ui_queue
        self.input_queue = input_queue
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._backend_ok = False
        self._sim_index = 0
        self._sim_phase_start = 0.0

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-backend", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    # ── Backend setup (imports are lazy so the UI never hard-depends) ───────

    def _import_backend(self) -> bool:
        """Import the real JARVIS modules. Returns True on success."""
        try:
            import config  # noqa: F401
            from brain.memory import ConversationMemory  # noqa: F401
            from brain.ollama_client import (  # noqa: F401
                OllamaClient,
            )
            from brain.router import Intent, IntentRouter  # noqa: F401
            from commands.registry import CommandRegistry  # noqa: F401
            from engine.microphone import MicrophoneManager  # noqa: F401
            from engine.stt import STTEngine  # noqa: F401
            from engine.tts import TTSEngine  # noqa: F401
            import requests  # noqa: F401
            return True
        except Exception:
            return False

    # ── Main loop ──────────────────────────────────────────────────────────

    def _run(self) -> None:
        self._backend_ok = self._import_backend()
        if self._backend_ok:
            self._run_backend()
        else:
            self._run_simulation()

    def _run_backend(self) -> None:
        from brain.memory import ConversationMemory
        from brain.ollama_client import OllamaClient
        from brain.router import IntentRouter
        from commands.registry import CommandRegistry
        from engine.microphone import MicrophoneManager
        from engine.stt import STTEngine
        from engine.tts import TTSEngine

        import config

        # Build the pipeline once; shared by both mic and text paths.
        self.memory = ConversationMemory()
        self.ollama = OllamaClient()
        self.router = IntentRouter()
        self.commands = CommandRegistry()
        self.tts = TTSEngine(rate=config.tts_config.RATE)
        mic = MicrophoneManager()
        self.mic = mic
        self.stt = STTEngine(mic)

        # Probe connectivity (non-blocking, never raises).
        self.state.mic_available = mic.is_available()
        try:
            self.state.ollama_connected = self.ollama.is_available()
        except Exception:
            self.state.ollama_connected = False
        self.ui_queue.put({"event": "stats", "cpu": 0.0, "ram": 0.0})

        if mic.is_available():
            try:
                self.stt.vad.calibrate()
            except Exception:
                pass

        greeting = (
            f"Good day, {config.jarvis_config.OWNER}. JARVIS is online and ready. "
            "How can I help you?"
        )
        self._respond(self.tts, greeting, from_boot=True)

        while not self._stop.is_set():
            self._drain_input_queue()

            if not self.state.mic_enabled:
                if self.state.get_status() != IDLE:
                    self.state.set_status(IDLE)
                    self.ui_queue.put(status_event(IDLE))
                time.sleep(0.05)
                continue

            if not mic.is_available():
                time.sleep(0.5)
                continue

            self.state.set_status(LISTENING)
            self.ui_queue.put(status_event(LISTENING))
            self.ui_queue.put(module_event("MICROPHONE", True))

            text = self.stt.listen()

            if self._stop.is_set():
                break
            if text:
                self._process_text(text)

            self.state.set_status(IDLE)
            self.ui_queue.put(status_event(IDLE))

    def _drain_input_queue(self) -> None:
        while True:
            try:
                text = self.input_queue.get_nowait()
            except queue.Empty:
                return
            text = text.strip()
            if not text:
                continue
            self._process_text(text)

    def _process_text(self, text: str) -> None:
        """Route + answer a single utterance (shared by mic and typed input)."""
        from brain.router import Intent

        self.state.current_input = text
        self.state.push_conversation("user", text)
        self.ui_queue.put({"event": "input", "text": text})

        self.state.set_status(THINKING)
        self.ui_queue.put(status_event(THINKING))
        self.ui_queue.put(module_event("STT", True))
        self.ui_queue.put(module_event("ROUTER", True))

        intent, cleaned = self.router.route(text)

        if intent == Intent.EXIT:
            self._respond(self.tts, "Goodbye, see you soon.")
            self.ui_queue.put(status_event(SHUTDOWN))
            self.stop()
            return

        if intent == Intent.CLEAR_MEMORY:
            self.memory.clear()
            self.state.clear_conversation()
            self.ui_queue.put({"event": "memory_cleared"})
            self._respond(self.tts, "I've cleared our conversation memory.")
            self.ui_queue.put(module_event("STT", False))
            self.ui_queue.put(module_event("ROUTER", False))
            return

        if intent == Intent.COMMAND:
            failed = False
            try:
                response = self.commands.execute(cleaned)
            except Exception:
                failed = True
                response = "Sorry, I couldn't complete that command."
            self.state.add_command_history(cleaned)
            if failed:
                self.state.set_status(ERROR)
                self.ui_queue.put(status_event(ERROR))
            self._respond(self.tts, response)
            self.ui_queue.put(module_event("STT", False))
            self.ui_queue.put(module_event("ROUTER", False))
            return

        # AI_QUESTION — the default for everything else.
        failed = False
        try:
            self.memory.add_user_message(cleaned)
            t0 = time.perf_counter()
            response = self.ollama.ask(cleaned, self.memory)
            self.state.response_time = time.perf_counter() - t0
            self.state.ollama_connected = True
            if not response:
                raise RuntimeError("empty provider response")
        except Exception:
            self.state.ollama_connected = False
            failed = True
            response = (
                "My AI brain is offline right now. Please start Ollama "
                "and I'll be back to normal."
            )
        self.state.current_intent = "AI_QUESTION"
        self.state.confidence = 0.94
        self.state.route = "Qwen3 Brain"
        self.state.thinking_progress = 1.0
        if failed:
            # Surface the failure to the UI (red ERROR state).
            self.state.set_status(ERROR)
            self.ui_queue.put(status_event(ERROR))
        self._respond(self.tts, response)
        self.ui_queue.put(module_event("STT", False))
        self.ui_queue.put(module_event("ROUTER", False))

    def _respond(self, tts, response: str, from_boot: bool = False) -> None:
        self.state.current_response = response
        self.state.push_conversation("assistant", response)
        if hasattr(self, "memory") and not from_boot:
            try:
                self.memory.add_assistant_message(response)
                self.state.memory_turns = max(1, len(self.memory) // 2)
            except Exception:
                pass
        self.ui_queue.put({"event": "response", "text": response})

        if not from_boot:
            self.state.set_status(SPEAKING)
            self.ui_queue.put(status_event(SPEAKING))
            self.ui_queue.put(module_event("TTS", True))

        try:
            tts.speak(response)
        except Exception:
            pass

        self.state.set_status(IDLE)
        self.ui_queue.put(status_event(IDLE))
        self.ui_queue.put(module_event("TTS", False))

    # ── Simulation fallback (no backend available) ──────────────────────────

    def _run_simulation(self) -> None:
        self.state.mic_available = True
        self.state.ollama_connected = False
        self.ui_queue.put({"event": "log", "text": "Backend unavailable — running UI demo mode."})

        demos = [
            ("What time is it?", "It's 3:42 PM, sir. Right on schedule."),
            ("Tell me a joke.", "Why did the neural network cross the road? To train on the other side."),
            ("What is quantum computing?", "Quantum computing uses qubits and superposition to solve problems far faster than classical machines."),
            ("Open YouTube.", "Opening YouTube for you now, sir."),
        ]

        def _fake_speak(text: str, duration: float) -> None:
            self.state.current_response = text
            self.state.push_conversation("assistant", text)
            self.ui_queue.put({"event": "response", "text": text})
            self.state.set_status(SPEAKING)
            self.ui_queue.put(status_event(SPEAKING))
            self.ui_queue.put(module_event("TTS", True))
            self._simulate_audio_wave(duration)
            self.state.set_status(IDLE)
            self.ui_queue.put(status_event(IDLE))
            self.ui_queue.put(module_event("TTS", False))

        greeting = "Good day, Sir. JARVIS is online in demo mode. How can I help you?"
        self.state.current_response = greeting
        self.state.push_conversation("assistant", greeting)
        self.ui_queue.put({"event": "response", "text": greeting})
        self._simulate_audio_wave(2.0)

        while not self._stop.is_set():
            self._drain_sim_input()

            if not self.state.mic_enabled:
                if self.state.get_status() != IDLE:
                    self.state.set_status(IDLE)
                    self.ui_queue.put(status_event(IDLE))
                time.sleep(0.05)
                continue

            self.state.set_status(LISTENING)
            self.ui_queue.put(status_event(LISTENING))
            self.ui_queue.put(module_event("MICROPHONE", True))
            self._simulate_audio_wave(2.5)

            if self._stop.is_set():
                break

            text, reply = demos[self._sim_index % len(demos)]
            self._sim_index += 1
            self.state.current_input = text
            self.state.push_conversation("user", text)
            self.ui_queue.put({"event": "input", "text": text})

            self.state.set_status(THINKING)
            self.ui_queue.put(status_event(THINKING))
            self.ui_queue.put(module_event("STT", True))
            self.ui_queue.put(module_event("ROUTER", True))
            self.ui_queue.put(module_event("OLLAMA", True))
            time.sleep(1.2)
            self.ui_queue.put(module_event("OLLAMA", False))
            self.ui_queue.put(module_event("STT", False))
            self.ui_queue.put(module_event("ROUTER", False))

            self.state.response_time = 1.2
            self.state.confidence = 0.93
            _fake_speak(reply, 2.8)

            self.state.set_status(IDLE)
            self.ui_queue.put(status_event(IDLE))

    def _drain_sim_input(self) -> None:
        """Handle typed input in simulation mode with a canned reply."""
        while True:
            try:
                text = self.input_queue.get_nowait()
            except queue.Empty:
                return
            text = text.strip()
            if not text:
                continue
            self.state.current_input = text
            self.state.push_conversation("user", text)
            self.ui_queue.put({"event": "input", "text": text})
            self.state.set_status(THINKING)
            self.ui_queue.put(status_event(THINKING))
            time.sleep(0.8)
            self.state.response_time = 0.8
            self.state.confidence = 0.91
            reply = (
                f"Demo mode: I heard \"{text}\". Connect the backend to get "
                "real answers from Qwen3."
            )
            self.state.current_response = reply
            self.state.push_conversation("assistant", reply)
            self.ui_queue.put({"event": "response", "text": reply})
            self.state.set_status(SPEAKING)
            self.ui_queue.put(status_event(SPEAKING))
            self._simulate_audio_wave(2.0)
            self.state.set_status(IDLE)
            self.ui_queue.put(status_event(IDLE))

    def _simulate_audio_wave(self, duration: float) -> None:
        """Drive the waveform with a synthetic voice-like signal."""
        start = time.perf_counter()
        phase = 0.0
        while time.perf_counter() - start < duration and not self._stop.is_set():
            elapsed = time.perf_counter() - start
            envelope = np.sin(np.pi * elapsed / duration) ** 2
            voice = 0.55 + 0.45 * np.sin(2 * np.pi * 3.2 * elapsed)
            wobble = 0.3 * np.sin(2 * np.pi * 0.7 * elapsed + phase)
            level = max(0.0, envelope * (0.35 + 0.65 * voice) + wobble * envelope)
            self.state.set_audio_level(float(min(1.0, level)))
            phase += 0.05
            time.sleep(1 / 60)


# ── System stats thread ───────────────────────────────────────────────────────

class SystemStatsThread(threading.Thread):
    """Samples CPU + RAM every ``interval`` seconds and pushes a stats event."""

    def __init__(self, state: JARVISState, ui_queue: queue.Queue, interval: float = 2.0) -> None:
        super().__init__(name="jarvis-stats", daemon=True)
        self.state = state
        self.ui_queue = ui_queue
        self.interval = interval
        self._stop = threading.Event()
        self._psutil = None
        try:
            import psutil  # noqa: F401

            self._psutil = psutil
        except Exception:
            self._psutil = None

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> None:
        while not self._stop.wait(self.interval):
            cpu = 0.0
            ram = 0.0
            if self._psutil is not None:
                try:
                    cpu = self._psutil.cpu_percent(interval=None)
                    ram = self._psutil.virtual_memory().percent
                except Exception:
                    pass
            else:
                cpu = 15 + (time.time() % 20)
                ram = 30 + (time.time() % 15)
            self.state.cpu_usage = cpu
            self.state.ram_usage = ram
            self.ui_queue.put(stats_event(cpu, ram))


# ── Audio level monitor thread ────────────────────────────────────────────────

class AudioMonitorThread(threading.Thread):
    """Reads microphone RMS level and pushes it into the shared state.

    Uses sounddevice's InputStream (already a backend dependency). If no mic
    is present, the thread simply idles and the waveform stays flat.
    """

    def __init__(self, state: JARVISState) -> None:
        super().__init__(name="jarvis-audio", daemon=True)
        self.state = state
        self._stop = threading.Event()
        self._stream = None

    def stop(self) -> None:
        self._stop.set()

    def _callback(self, indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        data = indata[:, 0]
        rms = float(np.sqrt(np.mean(np.square(data.astype(np.float64)))) + 1e-9)
        peak = float(np.max(np.abs(data))) if len(data) else 0.0
        level = min(1.0, rms * 6.0 + peak * 1.5)
        self.state.set_audio_level(level)

    def run(self) -> None:
        try:
            import sounddevice as sd

            device = None
            try:
                device = sd.default.device[0]
            except Exception:
                device = None
            self._stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype="float32",
                blocksize=512,
                device=device,
                callback=self._callback,
            )
            self._stream.start()
            while not self._stop.wait(0.05):
                pass
        except Exception:
            # No microphone / device busy — fall back to a gentle idle signal.
            while not self._stop.wait(1 / 30):
                current = self.state.get_audio_level()
                self.state.set_audio_level(max(0.0, current * 0.9))
        finally:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass