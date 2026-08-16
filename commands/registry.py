"""
Command registry — the single entry point for all local commands.
 
[FIX m2] Changed confirmation tokens from hex to 4-digit numeric codes
         that Whisper can recognize via voice.
[FIX m5] Added __all__ exports.
"""
 
import inspect
import random  # [FIX m2] For numeric codes
import re
import threading
import time
from dataclasses import dataclass, field
 
from commands import system_commands, time_commands
from commands.time_commands import DATE_RE, TIME_RE
from config import jarvis_config
 
# ── Command patterns ────────────────────────────────────────────────────────
 
# open / launch / start / run / go to / visit / show + target.
_OPEN_TARGET_PATTERNS = [
    (
        r"\b(?:open|launch|start|run|go\s+to|visit|show)\s+"
        r"(?:the\s+|my\s+)?\S+"
    ),
]
 
_STATUS_PATTERNS = [
    (
        r"\b(system status|computer status|pc status|system info|"
        r"system information|system health|how is my computer|"
        r"how is the computer)\b"
    ),
]
 
_VOLUME_PATTERNS = [
    r"\b(volume (up|down|mute|unmute))\b",
    r"\b(increase|raise|decrease|lower|turn (up|down))\s+(the\s+)?volume\b",
    r"\bset\s+(the\s+)?volume\s+to\s+\d{1,3}\s*(percent|%)?\b",
    r"\b(mute|unmute)\s+(the\s+)?(volume|sound|audio)\b",
]
 
_LOCK_PATTERNS = [
    r"\block\s+(my\s+|the\s+)?(computer|pc|laptop|system|machine|screen)\b",
]
 
# Require a computer noun so "abort shutdown" never matches this.
_POWER_PATTERNS = [
    (
        r"\b(shut ?down|power off|restart|reboot|sleep|hibernate)\s+"
        r"(my\s+|the\s+)?(computer|pc|system|laptop|machine)\b"
    ),
]
 
_ABORT_SHUTDOWN_PATTERNS = [
    r"\b(abort|cancel|stop)\s+shut ?down\b",
]
 
__all__ = [
    "PERMISSION_SAFE",
    "PERMISSION_CONFIRM",
    "PERMISSION_BLOCKED",
    "Command",
    "CommandResult",
    "PendingConfirmation",
    "CommandRegistry",
]
 
PERMISSION_SAFE = "safe"
PERMISSION_CONFIRM = "confirm"
PERMISSION_BLOCKED = "blocked"
 
OWNER = jarvis_config.OWNER
CONFIRMATION_TIMEOUT = jarvis_config.CONFIRMATION_TIMEOUT
CONFIRMATION_REQUIRE_TOKEN = jarvis_config.CONFIRMATION_REQUIRE_TOKEN
 
 
@dataclass
class Command:
    """A trusted, registered action JARVIS may perform."""
 
    name: str
    patterns: list[str]
    handler: object
    permission: str = PERMISSION_SAFE
    description: str = ""
    confirm_prompt: str = ""
    _compiled: list = field(default_factory=list, repr=False)
 
    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]
 
    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self._compiled)
 
    def execute(self, text: str) -> str:
        try:
            params = inspect.signature(self.handler).parameters
        except (TypeError, ValueError):
            params = {}
        if params:
            return str(self.handler(text))
        return str(self.handler())
 
 
@dataclass
class CommandResult:
    """What execute_with_meta() returns."""
 
    command: Command
    response: str = ""
 
    @property
    def name(self) -> str:
        return self.command.name
 
    @property
    def permission(self) -> str:
        return self.command.permission
 
    @property
    def confirm_prompt(self) -> str:
        return self.command.confirm_prompt
 
    @property
    def needs_confirmation(self) -> bool:
        return self.command.permission == PERMISSION_CONFIRM
 
    def execute(self, text: str) -> str:
        return self.command.execute(text)
 
 
class PendingConfirmation:
    """
    Tracks a CONFIRM-permission command awaiting an explicit "yes".
 
    [FIX m2] Changed token from secrets.token_hex(4) to 4-digit numeric
    code (1000-9999) that Whisper can recognize via voice input.
    """
 
    def __init__(
        self,
        result: CommandResult,
        original_text: str,
        timeout: float | None = None,
        require_token: bool | None = None,
    ) -> None:
        self.result = result
        self.original_text = original_text
        self.timeout = (
            timeout if timeout is not None else float(CONFIRMATION_TIMEOUT)
        )
        # [FIX m2] Use 4-digit numeric code instead of hex
        self.token = str(random.randint(1000, 9999))
        self.require_token = (
            require_token if require_token is not None
            else bool(CONFIRMATION_REQUIRE_TOKEN)
        )
        self._created = time.monotonic()
        self._lock = threading.Lock()
        self._consumed = False
 
    @property
    def prompt(self) -> str:
        """The spoken confirmation prompt."""
        base = self.result.confirm_prompt
        if self.require_token:
            return (
                f"{base} Say the code {self.token} to confirm, "
                "or say no to cancel."
            )
        return base
 
    @property
    def is_expired(self) -> bool:
        if self.timeout <= 0:
            return False
        return (time.monotonic() - self._created) > self.timeout
 
    def confirm(self, decision: str, token: str | None = None) -> bool:
        """True when the reply authorizes execution."""
        if self.is_expired or self._consumed:
            return False
        
        if decision == "no":
            return False
        
        if self.require_token:
            # [FIX m2] Token must match the numeric code
            return token == self.token
        
        return decision == "yes"
 
    def take(self, decision: str, token: str | None = None) -> CommandResult | None:
        """Consume the confirmation and return the command if authorized."""
        with self._lock:
            if self._consumed:
                return None
            if not self.confirm(decision, token):
                return None
            self._consumed = True
            return self.result
 
 
class CommandRegistry:
    """Registry of all available commands."""

    def __init__(self):
        self.system = system_commands.SystemCommands()
        self._commands: list[Command] = []
        self._register_defaults()

    def _register_defaults(self):
        """Register built-in commands."""
        # Time commands
        self._commands.append(Command(
            name="time",
            patterns=[TIME_RE.pattern],
            handler=time_commands.get_time,
            description="Tell the current time",
        ))
        
        self._commands.append(Command(
            name="date",
            patterns=[DATE_RE.pattern],
            handler=time_commands.get_date,
            description="Tell the current date",
        ))
        
        # Open apps / websites / folders
        self._commands.append(Command(
            name="open_target",
            patterns=_OPEN_TARGET_PATTERNS,
            handler=self.system._open_target,
            description="Open an app, website, or folder",
        ))
        
        # System commands
        self._commands.append(Command(
            name="screenshot",
            patterns=[r"\bscreenshot\b"],
            handler=self.system.take_screenshot,
            description="Take a screenshot",
        ))
        
        self._commands.append(Command(
            name="system_status",
            patterns=_STATUS_PATTERNS,
            handler=self.system.system_status,
            description="Show system status",
        ))
        
        self._commands.append(Command(
            name="volume",
            patterns=_VOLUME_PATTERNS,
            handler=self.system.volume_control,
            description="Adjust the master volume",
        ))
        
        self._commands.append(Command(
            name="lock_screen",
            patterns=_LOCK_PATTERNS,
            handler=self.system.lock_screen,
            description="Lock the computer",
        ))
        
        # Power commands (require confirmation)
        self._commands.append(Command(
            name="power",
            patterns=_POWER_PATTERNS,
            handler=self.system.power_action,
            permission=PERMISSION_CONFIRM,
            confirm_prompt=(
                f"{OWNER}, that will close your current session. "
                "Do you want me to continue?"
            ),
            description="Shut down, restart, or sleep the computer",
        ))
        
        self._commands.append(Command(
            name="abort_shutdown",
            patterns=_ABORT_SHUTDOWN_PATTERNS,
            handler=self.system.abort_shutdown,
            description="Cancel scheduled shutdown",
        ))
 
    def find(self, text: str) -> Command | None:
        """Find a command matching the text."""
        for cmd in self._commands:
            if cmd.matches(text):
                return cmd
        return None
 
    def execute(self, text: str) -> str | None:
        """Execute a command if found. Returns None for CONFIRM commands."""
        cmd = self.find(text)
        if cmd is None:
            return None
        
        if cmd.permission == PERMISSION_CONFIRM:
            return None  # Caller must handle confirmation flow
        
        return cmd.execute(text)
 
    def execute_with_meta(self, text: str) -> CommandResult | None:
        """Execute and return full metadata."""
        cmd = self.find(text)
        if cmd is None:
            return None
        
        if cmd.permission == PERMISSION_CONFIRM:
            return CommandResult(command=cmd)
        
        response = cmd.execute(text)
        return CommandResult(command=cmd, response=response)
 