"""
utils/terminal_ui.py — Terminal dashboard using Rich library.
 
[FIX M2] Created this missing module that di.py imports.
Provides a live terminal dashboard showing state, last text, current model,
and system metrics. Handles missing rich gracefully.
"""
 
import threading
from typing import Optional
 
from utils.logger import get_logger
 
__all__ = ["TerminalUI"]
 
logger = get_logger("terminal_ui")
 
# Try to import rich, but handle gracefully if missing
try:
    from rich.console import Console
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    logger.debug("Rich library not available. Terminal UI will be simplified.")
 
 
class TerminalUI:
    """
    Live terminal dashboard for JARVIS.
    
    Shows:
    - Current state (listening, processing, speaking)
    - Last transcribed text
    - Current AI model
    - System metrics (optional)
    
    Falls back gracefully when rich is not installed.
    """
 
    def __init__(self, debug: bool = False):
        self.debug = debug
        self._state = "idle"
        self._last_text = ""
        self._model = ""
        self._metrics: dict = {}
        self._lock = threading.Lock()
        self._live: Optional["Live"] = None
        self._running = False
        
        if RICH_AVAILABLE:
            self._console = Console()
        else:
            self._console = None
 
    def start(self) -> None:
        """Start the live display."""
        if not RICH_AVAILABLE:
            return
        
        self._running = True
        self._live = Live(
            self._render(),
            console=self._console,
            refresh_per_second=2,
            transient=True,
        )
        try:
            self._live.start()
        except Exception as e:
            logger.debug(f"Could not start live display: {e}")
            self._live = None
 
    def stop(self) -> None:
        """Stop the live display."""
        self._running = False
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None
 
    def update_state(self, state: str) -> None:
        """Update the current state (listening, processing, speaking)."""
        with self._lock:
            self._state = state
        self._refresh()
 
    def update_text(self, text: str) -> None:
        """Update the last transcribed text."""
        with self._lock:
            self._last_text = text[:100] + "..." if len(text) > 100 else text
        self._refresh()
 
    def update_model(self, model: str) -> None:
        """Update the current AI model name."""
        with self._lock:
            self._model = model
        self._refresh()
 
    def update_metrics(self, **kwargs) -> None:
        """Update system metrics (cpu, memory, etc.)."""
        with self._lock:
            self._metrics.update(kwargs)
        self._refresh()
 
    def _refresh(self) -> None:
        """Refresh the live display."""
        if self._live and self._running:
            try:
                self._live.update(self._render())
            except Exception:
                pass
 
    def _render(self):
        """Render the dashboard panel."""
        if not RICH_AVAILABLE:
            return ""
        
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan")
        table.add_column("Value", style="white")
        
        # State with color
        state_colors = {
            "listening": "green",
            "processing": "yellow",
            "speaking": "blue",
            "idle": "dim",
        }
        state_color = state_colors.get(self._state, "white")
        state_text = Text(self._state.upper(), style=f"bold {state_color}")
        table.add_row("State:", state_text)
        
        # Last text
        if self._last_text:
            table.add_row("Last:", Text(self._last_text, style="italic"))
        
        # Model
        if self._model:
            table.add_row("Model:", Text(self._model, style="magenta"))
        
        # Metrics
        if self._metrics:
            for key, value in self._metrics.items():
                table.add_row(f"{key}:", str(value))
        
        return Panel(
            table,
            title="[bold blue]JARVIS[/bold blue]",
            border_style="blue",
        )
 
    def print(self, message: str, style: str = "") -> None:
        """Print a message to the console."""
        if RICH_AVAILABLE and self._console:
            if self._live:
                self._live.console.print(message, style=style)
            else:
                self._console.print(message, style=style)
        else:
            print(message)
 
    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.print(f"[ERROR] {message}", style="bold red")
 
    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.print(f"[OK] {message}", style="bold green")
 
    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        self.print(f"[WARN] {message}", style="yellow")
 