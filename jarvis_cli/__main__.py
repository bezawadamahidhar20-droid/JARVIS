"""
jarvis_cli/__main__.py — entry point for `python -m jarvis_cli`.

Also executed directly by the jarvis.cmd launcher, so it bootstraps
sys.path to the repository root before importing the package.
"""

import os
import sys

# Make the repository root importable no matter what the CWD is.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jarvis_cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
