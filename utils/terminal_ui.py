"""JARVIS terminal dashboard.

A single persistent, auto-refreshing dashboard built on ``rich.live.Live``.
It is a *presentation layer only*: it never touches the mic, Whisper, the
router, Ollama, or Piper TTS, and it never re-processes anything.  The main
loop pushes *state* (``set_state``), *messages* (``add_message``), *component
status* (``set_component``) and *metrics* (``set_metric``); the dashboard
re-renders from that state on a background refresh thread.

Everything shown is real data:

* audio level        → actual RMS of the latest mic frame (``live_rms``)
* latency values     → stage timings pushed by ``main.py``
* session stats      → counters maintained by ``main.py``

The UI works in legacy Windows consoles too: Unicode box-drawing symbols are
used when the terminal supports them, otherwise plain ASCII is substituted so
the dashboard never crashes with ``UnicodeEncodeError``.
"""

import os
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

from rich import box
from rich.console import Console, Group, RenderResult, RenderableType
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text
from rich.layout import Layout

JARVIS_LOGO: str = (
    "       _____    ____ _    ___________\n"
    "      / /   |  / __ \\ |  / /  _/ ___/\n"
    " __  / / /| | / /_/ / | / // / \\__ \\ \n"
    "/ /_/ / ___ |/ _, _/| |/ // / ___/ / \n"
    "\\____/_/  |_/_/ |_| |___/___//____/  "
)


def supports_unicode() -> bool:
    """Best-effort check whether the active console can print Unicode."""
    enc: str = ""
    try:
        enc = (getattr(sys.stdout, "encoding", "") or "").lower()
    except Exception:
        pass
    if "utf" in enc or enc in ("cp65001", "utf_8"):
        return True
    env_ok = os.environ.get("WT_SESSION") or os.environ.get("TERM_PROGRAM")
    return bool(env_ok)


class _Dashboard:
    """Renderable that builds the whole layout fresh on every render.

    Because ``Live`` re-renders this object on its refresh thread, the spinner
    animates and the live audio level bar updates even while the main thread
    is blocked in a blocking call (mic capture, Whisper, TTS).
    """

    def __init__(self, ui: "TerminalUI") -> None:
        self.ui = ui

    def __rich_console__(
        self, console: Console, options
    ) -> RenderResult:
        # Yield (not return!): rich iterates the result of __rich_console__,
        # and Layout has no __iter__ — a returned Layout falls back to the
        # legacy sequence protocol (layout[0], layout[1], ...) and crashes
        # with KeyError('No layout with name 0'). Yielding is the rich
        # contract and lets console.render handle the Layout recursively.
        yield self.ui._build_layout()  # noqa: SLF001


class TerminalUI:
    """Persistent Rich dashboard for the JARVIS assistant."""

    def __init__(self, debug: bool = False) -> None:
        self.debug = debug
        self.unicode = supports_unicode()

        if self.unicode:
            self._ok = "✓"
            self._err = "✗"
            self._load = "◌"
            self._off = "○"
            self._arrow = "»"
            self._dash = "—"
            self._bullet = "•"
            self._spinner_name = "dots"
            self._bar_on = "█"
            self._bar_off = "░"
            self._box = box.ROUNDED
        else:
            self._ok = "+"
            self._err = "X"
            self._load = "*"
            self._off = "o"
            self._arrow = ">"
            self._dash = "-"
            self._bullet = "."
            self._spinner_name = "line"
            self._bar_on = "#"
            self._bar_off = "."
            self._box = box.ASCII

        self.console = Console(highlight=False)

        # Live state machine.
        self._state = "starting"
        self._state_meta = ""
        self._spinner: Spinner = Spinner(self._spinner_name, text="", style="cyan")

        # Components: name -> ("ok" | "error" | "off", detail).
        self._components: Dict[str, tuple] = {
            "Microphone": ("off", "—"),
            "Whisper": ("off", "—"),
            "Router": ("off", "—"),
            "Ollama": ("off", "—"),
            "TTS": ("off", "—"),
        }

        # Conversation: (kind, text) — kinds: user, jarvis, command, ai, notice.
        self._messages: Deque[tuple] = deque(maxlen=24)

        # Session stats (real counters maintained by main.py).
        self._requests = 0
        self._commands = 0
        self._questions = 0
        self._latencies: list = []
        self._session_start = time.time()

        # Performance: label -> seconds for the most recent turn.
        self._perf: Dict[str, float] = {}

        # Debug log lines (only kept/rendered in debug mode).
        self._debug_lines: Deque[str] = deque(maxlen=8)

        # Real audio level source (set via ``set_recorder``).
        self._recorder: Any = None

        self._live: Optional[Live] = None
        self._dashboard = _Dashboard(self)

    # ── Sink callback for utils/logger ────────────────────────────────────────
    def log_sink(self, level: str, payload: Any) -> None:
        """Callback wired to ``logger.set_sink``.

        * ``report`` events feed the performance panel (stage -> seconds).
        * ``error``/``warning`` appear as notices in the console area.
        * ``info``/``status``/``ok`` are shown only in debug mode.
        """
        if level == "report":
            label, elapsed = payload
            label = {"OLLAMA": "AI"}.get(label, label)
            self._perf[label] = float(elapsed)
            if label == "TOTAL":
                self._latencies.append(float(elapsed))
            return
        text = str(payload).strip()
        if not text:
            return
        if level in ("error", "warning"):
            self._messages.append((level, text))
        if self.debug:
            self._debug_lines.append(f"[{level}] {text}")

    # ── Public API used by main.py ────────────────────────────────────────────
    def start(self) -> None:
        self._live = Live(
            self._dashboard,
            console=self.console,
            screen=False,
            auto_refresh=True,
            refresh_per_second=10,
            vertical_overflow="crop",
        )
        self._live.start()

    def stop(self) -> None:
        """Stop the dashboard and print a session-summary panel."""
        if self._live is None:
            return
        try:
            self._live.update(self._summary_panel())
            self._live.stop()
        finally:
            self._live = None

    def refresh(self) -> None:
        if self._live is not None:
            self._live.refresh()

    def set_recorder(self, recorder: Any) -> None:
        self._recorder = recorder

    def set_component(self, name: str, status: str, detail: str = "—") -> None:
        self._components[name] = (status, detail)
        self.refresh()

    def set_state(self, state: str, meta: str = "") -> None:
        self._state = state
        if not self.unicode:
            meta = meta.replace("…", "...").replace("—", "-")
        self._state_meta = meta
        label = {
            "listening": "LISTENING",
            "thinking": "THINKING",
            "speaking": "SPEAKING",
            "error": "ERROR",
            "starting": "STARTING",
        }.get(state, state.upper())
        color = self._state_color()
        text = f" {label}"
        if meta:
            text += f"  •  {meta}"
        self._spinner = Spinner(self._spinner_name, text=text, style=color)
        self.refresh()

    def add_message(self, kind: str, text: str) -> None:
        self._messages.append((kind, text))
        if kind == "user":
            self._requests += 1
        elif kind == "command":
            self._commands += 1
        elif kind == "ai":
            self._questions += 1
        self.refresh()

    def set_metric(self, label: str, seconds: float) -> None:
        self._perf[label] = seconds
        if label == "TOTAL":
            self._latencies.append(seconds)
        self.refresh()

    # ── Rendering helpers ─────────────────────────────────────────────────────
    def _build_layout(self) -> RenderableType:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=8),
            Layout(name="body", ratio=1),
            Layout(name="footer", size=4),
        )
        layout["header"].update(self._header_panel())
        layout["body"].split_row(
            Layout(name="side", size=44),
            Layout(name="console", ratio=1),
        )
        layout["body"]["side"].split_column(
            Layout(name="system", size=14),
            Layout(name="session", size=8),
            Layout(name="performance", ratio=1),
        )
        layout["body"]["side"]["system"].update(self._system_panel())
        layout["body"]["side"]["session"].update(self._session_panel())
        layout["body"]["side"]["performance"].update(self._perf_panel())
        layout["body"]["console"].update(self._console_panel())
        layout["footer"].update(self._footer_panel())
        return layout

    def _header_panel(self) -> Panel:
        logo = Text(JARVIS_LOGO, style="cyan bold")
        tagline = Text(
            f"  LOCAL  {self._bullet}  AI  VOICE  ASSISTANT",
            style="dim",
        )
        tagline.append(
            f"  {self._arrow}  {self._state.upper()}",
            style=self._state_color(),
        )
        return Panel(
            Text("\n").join([logo, tagline]),
            box=self._box,
            border_style="cyan",
            title="JARVIS",
            title_align="left",
        )

    def _system_panel(self) -> Panel:
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold cyan")
        table.add_column()
        for name, (status, detail) in self._components.items():
            if status == "ok":
                mark = self._ok
                color = "green"
            elif status == "error":
                mark = self._err
                color = "red"
            else:
                mark = self._load
                color = "yellow"
            if detail in ("—", "-"):
                detail = self._dash
            label = Text(f"{mark} {name}", style=color)
            table.add_row(label, detail)
        return Panel(table, title="SYSTEM", box=self._box, border_style="dim")

    def _session_panel(self) -> Panel:
        elapsed = time.time() - self._session_start
        avg = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        stats = Table.grid(expand=True, padding=(0, 1))
        stats.add_column(style="bold", justify="right")
        stats.add_column(style="cyan")
        stats.add_row("Requests", f"{self._requests}")
        stats.add_row("Commands", f"{self._commands}")
        stats.add_row("Questions", f"{self._questions}")
        stats.add_row("Avg latency", f"{avg:.1f}s" if self._latencies else self._dash)
        stats.add_row("Session", f"{elapsed:.0f}s")
        return Panel(stats, title="SESSION", box=self._box, border_style="dim")

    def _perf_panel(self) -> Panel:
        rows = [("CAPTURE", "CAPTURE"), ("WHISPER", "WHISPER")]
        rows += [("AI", "AI"), ("TTS", "TTS"), ("TOTAL", "TOTAL")]
        table = Table.grid(expand=True, padding=(0, 1))
        table.add_column(style="bold", justify="right")
        table.add_column(style="cyan")
        for label, key in rows:
            val = self._perf.get(key)
            table.add_row(label, f"{val:.2f}s" if val is not None else self._dash)
        return Panel(table, title="PERFORMANCE", box=self._box, border_style="dim")

    def _console_panel(self) -> Panel:
        parts: list = []
        for kind, text in list(self._messages)[-8:]:
            if kind == "user":
                parts.append(Text(f"{self._arrow} YOU", style="yellow bold"))
                parts.append(Text("  " + text, style="white"))
            elif kind == "jarvis":
                parts.append(Text(f"{self._arrow} JARVIS", style="cyan bold"))
                parts.append(Text("  " + text, style="bright_white"))
            elif kind == "command":
                parts.append(Text(f"{self._ok} COMMAND  {text}", style="green"))
            elif kind == "ai":
                parts.append(Text(f"{self._ok} AI  {text}", style="blue"))
            elif kind == "error":
                parts.append(Text(f"{self._err} {text}", style="red"))
            elif kind == "warning":
                parts.append(Text(f"! {text}", style="yellow"))
            else:
                parts.append(Text(text, style="dim"))
        if not parts:
            parts.append(Text(f"  Standing by {self._dash} say something...", style="dim"))
        return Panel(
            Text("\n").join(parts),
            title="JARVIS CONSOLE",
            box=self._box,
            border_style="bright_blue",
            expand=True,
        )

    def _footer_panel(self) -> Panel:
        parts: list = []
        parts.append(self._spinner)

        # Real audio level bar (only meaningful while listening).
        parts.append(self._level_bar())

        if self.debug and self._debug_lines:
            parts.append(
                Text(" | ".join(self._debug_lines), style="dim", overflow="ellipsis")
            )
        return Panel(
            Group(*parts),
            box=self._box,
            border_style=self._state_color(),
        )

    def _level_bar(self) -> Text:
        width = 24
        rms = 0.0
        speaking = False
        if self._recorder is not None:
            try:
                rms = float(self._recorder.live_rms)
            except Exception:
                rms = 0.0
            try:
                speaking = bool(self._recorder.live_speech)
            except Exception:
                speaking = False
        # Scale RMS up to a comfortable ceiling; clamp to [0, 1].
        frac = min(1.0, rms / 0.15)
        filled = int(round(frac * width))
        bar = self._bar_on * filled + self._bar_off * (width - filled)
        text = Text("AUDIO  ")
        if speaking:
            text.append(bar, style="yellow")
            text.append("  SPEECH DETECTED", style="yellow bold")
        else:
            text.append(bar, style="green" if frac > 0.05 else "dim")
            text.append(f"  {rms:.4f}", style="dim")
        return text

    def _summary_panel(self) -> Panel:
        elapsed = time.time() - self._session_start
        avg = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        body = Text()
        body.append("SESSION COMPLETE\n", style="cyan bold")
        body.append(f"Requests:   {self._requests}\n", style="white")
        body.append(f"Commands:   {self._commands}\n", style="white")
        body.append(f"Questions:  {self._questions}\n", style="white")
        body.append(
            f"Avg latency: {avg:.1f}s" if self._latencies else f"Avg latency: {self._dash}",
            style="white",
        )
        body.append(f"\nSession time: {elapsed:.0f}s\n", style="white")
        body.append("\nJARVIS offline. Thank you for using me.", style="cyan")
        return Panel(body, title="JARVIS", box=self._box, border_style="cyan")

    def _state_color(self) -> str:
        return {
            "listening": "green",
            "thinking": "yellow",
            "speaking": "cyan",
            "error": "red",
            "starting": "cyan",
        }.get(self._state, "cyan")