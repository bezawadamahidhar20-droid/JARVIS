"""Demonstrate the command router without executing anything.

Execution is skipped so you can see the routing logic safely first.
Run with  --exec  to actually open the apps.
"""

import sys
import shutil

from commands.router import CommandRouter, _find_chrome

SAMPLES = (
    "open Notepad",
    "open Calculator",
    "open Chrome",
    "open File Explorer",
    "open Command Prompt",
    "what time is it?",
    "what is the date today?",
    "open chrome",
    "goodbye",
    "what is the capital of Japan",
)


def main():
    router = CommandRouter()
    do_exec = "--exec" in sys.argv

    if do_exec:
        if _find_chrome() is None:
            print("[!] Chrome not found — test other apps, or install Chrome.")
        for app in ("notepad", "explorer", "cmd"):
            which = shutil.which(app)
            print(f"[i] '{app}' resolvable: {which}")

    for sample in SAMPLES:
        name, _ = router.route(sample)
        print(f"{sample!r:45} -> {name}")

    if do_exec:
        print("\n[*] Executing test commands...")
        for sample in SAMPLES[:5] + SAMPLES[8:9]:
            name, _ = router.route(sample)
            if name:
                print(router.execute(name))


if __name__ == "__main__":
    main()