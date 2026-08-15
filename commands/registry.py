"""Command registry — the single entry point for all local commands.

The router decides an utterance is a ``COMMAND``; this registry decides
*which* command and executes it, returning what JARVIS should say.
"""

from commands import system_commands, time_commands
from commands.time_commands import DATE_RE, TIME_RE


class CommandRegistry:
    """Routes matched commands to time/date or system actions."""

    def __init__(self) -> None:
        self.system = system_commands.SystemCommands()

    def execute(self, user_input: str) -> str:
        """Execute *user_input* and return the spoken response string.

        If nothing matches a known command, a polite fallback message is
        returned so the caller always has something to say.
        """
        t = (user_input or "").strip()

        if TIME_RE.search(t):
            return time_commands.get_current_time()
        if DATE_RE.search(t):
            return time_commands.get_current_date()

        # Everything else (open X / screenshot) goes to SystemCommands.
        response = self.system.execute(t)
        if response:
            return response

        return "I understood that as a command, but I don't know how to do it."