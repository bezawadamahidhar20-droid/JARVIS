"""Rich terminal dashboard rendering tests (utils/terminal_ui.py).

Guards the pre-existing crash where every render of the live dashboard
raised ``KeyError: 'No layout with name 0'`` on a real (TTY) console,
flooding the main loop with "Recording error" and breaking Ctrl+C
shutdown.
"""

import io

from rich.console import Console

from utils.terminal_ui import TerminalUI


def _make_ui() -> TerminalUI:
    ui = TerminalUI(debug=False)
    ui.update_state("listening")
    ui.update_model("qwen3:8b")
    ui.update_text("user: hello there")
    ui.update_metrics(AI=1.2)
    return ui


def test_dashboard_renders_via_live_render_path():
    """The LiveRender path must not crash.

    ``LiveRender.__rich_console__`` renders the dashboard with
    ``console.render_lines(..., pad=False)``. A ``__rich_console__``
    that RETURNS a rich ``Layout`` object (instead of yielding it) made
    rich iterate the layout through the legacy sequence protocol
    (``layout[0]``, ``layout[1]``, ...), which raised
    ``KeyError('No layout with name 0')`` at every refresh.
    """
    ui = _make_ui()
    console = Console(force_terminal=True, width=80, height=30, file=io.StringIO())
    for height in (30, 15, 8, 5):
        lines = console.render_lines(
            ui._render(),  # noqa: SLF001  (dashboard is the render target)
            console.options.update_dimensions(80, height),
            pad=False,
        )
        assert lines


def test_dashboard_prints_via_console():
    """console.print of the dashboard (the FileProxy / render-hook path
    used while the Live is active) must not raise."""
    ui = _make_ui()
    console = Console(force_terminal=True, width=80, height=30, file=io.StringIO())
    console.print(ui._render())  # noqa: SLF001
    assert console.file.getvalue().strip()
