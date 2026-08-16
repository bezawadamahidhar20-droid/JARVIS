"""
di.py — DependencyContainer: a small service-locator / DI container.
 
[FIX M2] Updated _ui factory to import from utils.terminal_ui
(which now exists after fix M2).
"""
 
import logging
 
logger = logging.getLogger("di")
 
# Registry keys
MIC = "mic"
STT = "stt"
TTS = "tts"
MEMORY = "memory"
PROVIDER = "provider"
SEARCH = "search"
ROUTER = "router"
COMMANDS = "commands"
UI = "ui"
 
__all__ = [
    "DependencyContainer",
    "build_default_container",
    "MIC", "STT", "TTS", "MEMORY", "PROVIDER",
    "SEARCH", "ROUTER", "COMMANDS", "UI",
]
 
 
class DependencyContainer:
    """Lazily resolves components; overrides always win."""
 
    def __init__(self) -> None:
        self._factories: dict[str, callable] = {}
        self._overrides: dict[str, object] = {}
        self._instances: dict[str, object] = {}
 
    def register(self, key: str, factory: callable) -> None:
        self._factories[key] = factory
        self._instances.pop(key, None)
 
    def override(self, key: str, instance: object) -> None:
        self._overrides[key] = instance
        self._instances.pop(key, None)
 
    def has(self, key: str) -> bool:
        return key in self._overrides or key in self._factories
 
    def get(self, key: str):
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
 
    def audio_pipeline(self) -> tuple:
        return (self.get(MIC), self.get(STT), self.get(TTS))
 
    def brain(self) -> tuple:
        return (self.get(MEMORY), self.get(PROVIDER), self.get(SEARCH))
 
    def router(self):
        return self.get(ROUTER)
 
    def commands(self):
        return self.get(COMMANDS)
 
    def ui(self):
        return self.get(UI)
 
 
def build_default_container(debug: bool = False) -> DependencyContainer:
    """The production container with real engines."""
 
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
        # [FIX M2] Import from utils.terminal_ui which now exists
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