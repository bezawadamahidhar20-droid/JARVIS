"""Command registry — the single entry point for all local commands.

The router decides an utterance is a ``COMMAND``; this registry decides
*which* registered command it is and executes it, returning what JARVIS
should say.

Security model (see README):
  * Every OS action comes from a fixed, trusted :class:`Command` in this
    registry — never from LLM output.
  * Each command carries a permission level:

      SAFE    — execute immediately (open apps, websites, folders,
                time/date, screenshot, system status, volume, lock)
      CONFIRM — ask the user first, execute only after an explicit
                "yes" (shutdown / restart / sleep)
      BLOCKED — never registered; unknown utterances go to the AI or
                get a polite fallback, and are never executed

  * ``execute()`` (back-compat) never runs a CONFIRM command — it
    returns the confirmation prompt instead.  ``execute_with_meta()``
    returns full metadata so main.py can drive the confirmation flow.
"""

import inspect
import re
import secrets
import threading
import time
from dataclasses import dataclass, field

from commands import system_commands, time_commands
from commands.time_commands import DATE_RE, TIME_RE

# Permission levels.
PERMISSION_SAFE = "safe"
PERMISSION_CONFIRM = "confirm"
PERMISSION_BLOCKED = "blocked"

try:
    from config import jarvis_config

    OWNER = jarvis_config.OWNER
    CONFIRMATION_TIMEOUT = jarvis_config.CONFIRMATION_TIMEOUT
    CONFIRMATION_REQUIRE_TOKEN = jarvis_config.CONFIRMATION_REQUIRE_TOKEN
except Exception:
    OWNER = "Sir"
    CONFIRMATION_TIMEOUT = 30
    CONFIRMATION_REQUIRE_TOKEN = False


# ── Command patterns ────────────────────────────────────────────────────────

# open / launch / start / run / go to / visit / show + target.
# The pattern is deliberately broad (any target): the handler resolves
# against the safe WEBSITES/APPS/FOLDERS tables and politely reports
# unknown targets — so "open quantum-flux-device" never executes
# anything, it just gets an honest "I don't know how to open" reply.
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


@dataclass
class Command:
    """A trusted, registered action JARVIS may perform."""

    name: str
    patterns: list[str]
    handler: object  # callable(text) -> str
    permission: str = PERMISSION_SAFE
    description: str = ""
    confirm_prompt: str = ""
    _compiled: list = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.patterns]

    def matches(self, text: str) -> bool:
        return any(p.search(text) for p in self._compiled)

    def execute(self, text: str) -> str:
        """Run the handler, passing the utterance only if it takes one.

        Some handlers (screenshot, lock, system status) ignore the
        utterance entirely; the dispatch adapts to the signature.
        """
        try:
            params = inspect.signature(self.handler).parameters
        except (TypeError, ValueError):
            params = {}
        if params:
            return str(self.handler(text))
        return str(self.handler())


@dataclass
class CommandResult:
    """What execute_with_meta() returns: the matched command plus the
    spoken response (for SAFE commands) or the confirmation prompt."""

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
        """Run the command's handler (used after the user confirms)."""
        return self.command.execute(text)


class PendingConfirmation:
    """
    Tracks a CONFIRM-permission command awaiting an explicit "yes".

    Guarantees:
      * ``take()`` returns the pending action at most ONCE — repeated
        "yes" replies can never execute the command twice.
      * After ``timeout`` seconds the confirmation expires and
        ``take()`` returns None, so a stale confirmation can never be
        executed later.
      * All transitions are guarded by a lock, so concurrent inputs
        cannot race the pending state.
      * Each confirmation carries a random nonce (``token``). With
        ``require_token`` enabled (CONFIRMATION_REQUIRE_TOKEN=true) the
        user must echo that code back before the command executes, so a
        stray "yes" from a second process or an injected stdin write
        can never authorize a destructive action.

    The caller clears its reference on every terminal outcome
    (success, rejection, timeout, exception) — see main.py.
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
        # timeout <= 0 disables the expiry (not recommended).
        self.timeout = (
            timeout if timeout is not None else float(CONFIRMATION_TIMEOUT)
        )
        # Random nonce binding this specific confirmation to the reply.
        # secrets.token_hex(4) = 8 hex chars (~32 bits of entropy).
        self.token = secrets.token_hex(4)
        self.require_token = (
            require_token if require_token is not None
            else bool(CONFIRMATION_REQUIRE_TOKEN)
        )
        self._created = time.monotonic()
        self._lock = threading.Lock()
        self._consumed = False

    @property
    def prompt(self) -> str:
        """The spoken confirmation prompt, including the nonce when
        token confirmation is required."""
        base = self.result.confirm_prompt
        if self.require_token:
            return (
                f"{base} Say the code {self.token} to confirm, "
                "or say no."
            )
        return base

    @property
    def is_expired(self) -> bool:
        """True when the confirmation window has passed."""
        if self.timeout <= 0:
            return False
        return (time.monotonic() - self._created) > self.timeout

    def confirm(self, decision: str, token: str | None = None) -> bool:
        """
        True when the reply authorizes execution of the pending action.

        Args:
            decision: "yes" | "no" | "other" (from main._confirm_decision)
            token:    the code the user echoed back. Required when
                      ``require_token`` is on; ignored otherwise.

        * Without token confirmation: only an explicit "yes" authorizes.
        * With token confirmation: echoing the correct code authorizes
          (the code IS the affirmative); an explicit "no" still wins.

        Never returns True after the timeout or after ``take()`` has
        claimed the action.
        """
        if self.is_expired or self._consumed:
            return False
        if self.require_token:
            if not token or not secrets.compare_digest(
                token.strip().lower(), self.token.lower()
            ):
                return False
            return decision != "no"
        return decision == "yes"

    def take(self) -> tuple[CommandResult, str] | None:
        """
        Atomically claim the pending command if it is still valid.

        Returns (result, original_text) exactly once; every later call
        (and any call after the timeout) returns None.
        """
        with self._lock:
            if self._consumed or self.is_expired:
                return None
            self._consumed = True
            return self.result, self.original_text


# Static time/date commands (matched via the shared regexes, not patterns).
_TIME_COMMAND = Command(
    name="time",
    patterns=[],
    handler=lambda _t: time_commands.get_current_time(),
    description="Current time from the system clock",
)
_DATE_COMMAND = Command(
    name="date",
    patterns=[],
    handler=lambda _t: time_commands.get_current_date(),
    description="Today's date from the system clock",
)


class CommandRegistry:
    """Routes matched commands to the right handler.

    Time/date are handled first (they are matched by the shared regexes
    from commands/time_commands.py so the router and registry can never
    disagree about them). Everything else is matched against the
    registered Command table.
    """

    def __init__(self) -> None:
        self.system = system_commands.SystemCommands()
        self.commands: list[Command] = self._build_commands()

    # ── Command table ─────────────────────────────────────────

    def _build_commands(self) -> list[Command]:
        return [
            Command(
                name="open_target",
                patterns=_OPEN_TARGET_PATTERNS,
                handler=self.system._open_target,  # internal resolver
                permission=PERMISSION_SAFE,
                description="Open an app, website, or folder",
            ),
            Command(
                name="screenshot",
                patterns=[r"\bscreenshot\b"],
                handler=self.system.take_screenshot,
                permission=PERMISSION_SAFE,
                description="Take a screenshot",
            ),
            Command(
                name="system_status",
                patterns=_STATUS_PATTERNS,
                handler=self.system.system_status,
                permission=PERMISSION_SAFE,
                description="Report CPU, RAM, and disk health",
            ),
            Command(
                name="volume",
                patterns=_VOLUME_PATTERNS,
                handler=self.system.volume_control,
                permission=PERMISSION_SAFE,
                description="Adjust the master volume",
            ),
            Command(
                name="lock_screen",
                patterns=_LOCK_PATTERNS,
                handler=self.system.lock_screen,
                permission=PERMISSION_SAFE,
                description="Lock the Windows session",
            ),
            Command(
                name="power",
                patterns=_POWER_PATTERNS,
                handler=self.system.power_action,
                permission=PERMISSION_CONFIRM,
                description="Shut down, restart, or sleep the computer",
                confirm_prompt=(
                    f"{OWNER}, that will close your current session. "
                    "Do you want me to continue?"
                ),
            ),
            Command(
                name="abort_shutdown",
                patterns=_ABORT_SHUTDOWN_PATTERNS,
                handler=self.system.abort_shutdown,
                permission=PERMISSION_SAFE,
                description="Cancel a scheduled shutdown/restart",
            ),
        ]

    # ── Matching ──────────────────────────────────────────────

    def find(self, text: str) -> Command | None:
        """Return the first Command matching *text*, or None."""
        t = (text or "").strip()
        if not t:
            return None
        for cmd in self.commands:
            if cmd.matches(t):
                return cmd
        return None

    # ── Dispatch ──────────────────────────────────────────────

    def execute_with_meta(self, text: str) -> CommandResult | None:
        """
        Classify *text* and return full metadata (response + permission).

        Returns None when nothing matches (caller falls back politely).
        Time/date are resolved here via the shared regexes.
        """
        t = (text or "").strip()
        if not t:
            return None

        if TIME_RE.search(t):
            return CommandResult(_TIME_COMMAND, _TIME_COMMAND.execute(t))
        if DATE_RE.search(t):
            return CommandResult(_DATE_COMMAND, _DATE_COMMAND.execute(t))

        cmd = self.find(t)
        if cmd is None:
            return None

        if cmd.permission == PERMISSION_CONFIRM:
            # Do NOT execute — the caller must ask first and only run
            # after an explicit "yes".
            return CommandResult(cmd, "")

        return CommandResult(cmd, cmd.execute(t))

    def execute(self, user_input: str) -> str:
        """Back-compat: execute *user_input* and return the spoken reply.

        Safe commands run immediately. Confirmation commands return
        their confirmation prompt WITHOUT executing — so this entry
        point (used by the GUI and tests) can never trigger a shutdown
        without an explicit confirmation step.
        """
        result = self.execute_with_meta(user_input)
        if result is None:
            return "I understood that as a command, but I don't know how to do it."
        if result.needs_confirmation:
            return result.confirm_prompt
        return result.response
