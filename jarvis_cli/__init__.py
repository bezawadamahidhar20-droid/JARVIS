"""
jarvis_cli — the `jarvis` command.

Usage (from ANY directory after install):
    jarvis                 start JARVIS (voice mode)
    jarvis --text          text-only mode (no microphone)
    jarvis --debug         verbose debug logging on the console
    jarvis --benchmark     print per-stage latency report on exit
    jarvis --doctor        run the health check and print fixes
    jarvis --version       show the installed version
    jarvis --help          show this help
    jarvis --startup enable|disable   Windows auto-start at login
    jarvis --gui           launch the desktop GUI (PySide6 + OpenGL)

The repository root is prepended to sys.path at import time so the
flat modules (config, brain, engine, commands, utils) resolve no
matter what the current working directory is.
"""

import os
import sys

__version__ = "2.0.0"

# Make the repository root importable regardless of CWD.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.logger import get_logger  # noqa: E402

logger = get_logger("cli")


def _build_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="jarvis",
        description=(
            "JARVIS — a private, local AI voice assistant for Windows. "
            "Microphone → VAD → Faster-Whisper → Router → Ollama → Piper."
        ),
        epilog=(
            "Examples:\n"
            "  jarvis                start in voice mode\n"
            "  jarvis --text         chat without a microphone\n"
            "  jarvis --doctor       health check\n"
        ),
    )
    parser.add_argument(
        "--text", "--text-mode", action="store_true",
        help="run in text mode (no microphone needed)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="show DEBUG-level logging on the console",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="print a per-stage latency report when JARVIS exits",
    )
    parser.add_argument(
        "--doctor", action="store_true",
        help="run the health check and print fixes for any problems",
    )
    parser.add_argument(
        "--version", "-V", action="store_true",
        help="show the installed version and exit",
    )
    parser.add_argument(
        "--startup", metavar="ACTION", choices=("enable", "disable"),
        help="add JARVIS to Windows startup at login (enable/disable)",
    )
    parser.add_argument(
        "--gui", action="store_true",
        help="launch the desktop GUI (PySide6 + OpenGL)",
    )
    return parser


def _print_version() -> int:
    print(f"JARVIS version {__version__}")
    print("Local AI voice assistant for Windows.")
    return 0


def _run_gui() -> int:
    try:
        from jarvis_ui.ui_main import main as gui_main
    except ImportError:
        print(
            "[!] GUI dependencies missing. Install with: "
            "pip install -r jarvis_ui/requirements_ui.txt"
        )
        logger.debug("GUI import failed:", exc_info=True)
        return 1
    return gui_main()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.version:
        return _print_version()

    if args.doctor:
        from jarvis_cli.doctor import run_doctor

        return run_doctor(verbose=args.debug)

    if args.startup:
        from jarvis_cli.startup import handle_startup

        return handle_startup(args.startup)

    if args.gui:
        return _run_gui()

    # Default: run the assistant.
    from main import run_assistant

    return run_assistant(
        text_mode=args.text,
        debug=args.debug,
        benchmark=args.benchmark,
    )


if __name__ == "__main__":
    sys.exit(main())
