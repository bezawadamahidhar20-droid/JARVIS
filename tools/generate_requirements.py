"""tools/generate_requirements.py — regenerate requirements.txt from pyproject.toml.

pyproject.toml is the single source of truth for runtime dependencies
([project].dependencies). This script rewrites requirements.txt to match
exactly, so the two files can never drift again.

    python -m tools.generate_requirements
"""

import pathlib
import tomllib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent


def main() -> int:
    with open(_PROJECT_ROOT / "pyproject.toml", "rb") as f:
        data = tomllib.load(f)
    deps = data["project"]["dependencies"]
    header = (
        "# Generated from pyproject.toml [project].dependencies -- do not edit by hand.\n"
        "# Source of truth: pyproject.toml. Regenerate with:\n"
        "#   python -m tools.generate_requirements\n"
        "#\n"
    )
    target = _PROJECT_ROOT / "requirements.txt"
    target.write_text(header + "\n".join(deps) + "\n", encoding="utf-8")
    print(f"requirements.txt regenerated from pyproject.toml ({len(deps)} deps).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())