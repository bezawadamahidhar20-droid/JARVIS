"""
di.py — DependencyContainer: a small service-locator / DI container.

Why: main.py used to wire every subsystem directly in JARVIS.__init__,
which made the orchestrator a god-object and forced tests to reach
through the constructor. The container formalizes the wiring:

  * Every subsystem is built lazily by a registered factory — nothing
    is instantiated until it is first requested.
  * ``override(key, instance)`` swaps any component for a fake (tests,
    the GUI, alternate providers) without monkey-patching.
  * The JARVIS orchestrator only ever asks for interfaces — it never
    constructs concrete engines itself.

Usage::

    container = DependencyContainer()
    jarvis = JARVIS(container=container)          # production
    container.override("provider", FakeProvider())  # tests
"""

import logging

logger = logging.getLogger("di")

# Registry keys (module-level constants so both sides agree on names).
MIC = "mic"
STT = "stt"
TTS = "tts"
MEMORY = "memory"
PROVIDER = "provider"
SEARCH = "search"
ROUTER = "router"
COMMANDS = "commands"
UI = "ui"


class DependencyContainer:
    """Lazily resolves components; overrides always win."""

    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}
        self._overrides: dict[str, object] = {}
        self._instances: dict[str, object] = {}

    # ── Registration ──────────────────────────────────────────

    def register(self, key: str, factory: callable) -> None:
        """Register ``factory(container) -> component`` for *key*."""
        self._factories[key] = factory
        self._instances.pop(key, None)

    def override(self, key: str, instance: object) -> None:
        """Force *key* to resolve to *instance* (even None)."""
        self._overrides[key] = instance
        self._instances.pop(key, None)

    def has(self, key: str) -> bool:
        return key in self._overrides or key in self._factories

    # ── Resolution ────────────────────────────────────────────

    def get(self, key: str):
        """Resolve *key*: override > cached singleton > factory."""
        if key in self._overrides:
            return self._overrides[key]
        if key in self._instances:
            return self._instances[key]
        factory = self._factories.get(key)
        if factory is None:
            raise KeyError(f"No component registered for {key!r}")
        instance = factory(self)
        self._instances[key] = instance
        return instance

    # ── Grouped accessors (the audit's suggested shape) ───────

    def audio_pipeline(self) -> tuple:
        """(mic, stt, tts) — capture + speech pipeline."""
        return (self.get(MIC), self.get(STT), self.get(TTS))

    def brain(self) -> tuple:
        """(memory, provider, search) — the reasoning layer."""
        return (self.get(MEMORY), self.get(PROVIDER), self.get(SEARCH))

    def router(self):
        return self.get(ROUTER)

    def commands(self):
        return self.get(COMMANDS)

    def ui(self):
        return self.get(UI)


# ── Default wiring (production) ───────────────────────────────

def build_default_container(debug: bool = False) -> DependencyContainer:
    """The production container: real engines wired with .env config.

    Provider/search creation is resilient: an unreachable or
    misconfigured provider degrades to None (JARVIS keeps running on
    local commands); a genuine bug in provider construction re-raises
    instead of being silently swallowed.

    Args:
        debug: forwarded to the terminal dashboard (verbosity).
    """

    def _mic(c):
        from engine.microphone import MicrophoneManager

        return MicrophoneManager()

    def _stt(c):
        from engine.stt import STTEngine

        return STTEngine(c.get(MIC))

    def _tts(c):
        from config import tts_config
        from engine.tts import TTSEngine

        return TTSEngine(rate=tts_config.RATE)

    def _memory(c):
        from brain.memory import ConversationMemory

        return ConversationMemory()

    def _provider(c):
        from brain.exceptions import ProviderUnavailableError
        from brain.llm import create_provider

        try:
            return create_provider()
        except (ValueError, ProviderUnavailableError) as e:
            logger.warning("AI provider unavailable at startup: %s", e)
            return None
        except Exception:
            logger.exception("Unexpected error creating AI provider")
            raise

    def _search(c):
        from brain.search import create_search_provider

        try:
            return create_search_provider()
        except Exception as e:
            logger.warning("Web search unavailable: %s", e)
            return None

    def _router(c):
        from brain.router import IntentRouter

        return IntentRouter()

    def _commands(c):
        from commands.registry import CommandRegistry

        return CommandRegistry()

    def _ui(c):
        from utils.terminal_ui import TerminalUI

        return TerminalUI(debug=debug)

    container = DependencyContainer()
    container.register(MIC, _mic)
    container.register(STT, _stt)
    container.register(TTS, _tts)
    container.register(MEMORY, _memory)
    container.register(PROVIDER, _provider)
    container.register(SEARCH, _search)
    container.register(ROUTER, _router)
    container.register(COMMANDS, _commands)
    container.register(UI, _ui)
    return container
