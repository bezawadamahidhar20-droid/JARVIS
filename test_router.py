"""Demonstrate the command router without executing anything.

Execution is skipped so you can see the routing logic safely first.
Run with  --exec  to actually open the apps.

Exit phrases ("goodbye", "exit", "quit", ...) are intentionally NOT routed
here - they are handled by is_exit_phrase() in main.py, so the router must
never return "exit" (that would let words inside normal sentences shut down
JARVIS).
"""

import shutil
import sys

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
    "what is the capital of Japan",
)

# These must never be routed to a command (exit lives in main.py).
NON_ROUTED_PHRASES = (
    "goodbye",
    "exit",
    "quit",
    "what is an exit code?",
    "how do i quit vim?",
    "tell me about exit strategies",
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

    print("\n-- Exit-safety check (should all be -> None) --")
    failed = False
    for phrase in NON_ROUTED_PHRASES:
        name, _ = router.route(phrase)
        status = "OK" if name is None else "FAIL (routed to exit)"
        if name is not None:
            failed = True
        print(f"{phrase!r:45} -> {name}  [{status}]")

    if failed:
        print("\n[!] ERROR: the router routed an exit phrase. "
              "Exit handling must stay in main.py.")
        sys.exit(1)

    if do_exec:
        print("\n[*] Executing test commands...")
        for sample in SAMPLES[:5]:
            name, _ = router.route(sample)
            if name:
                print(router.execute(name))

    print("\n[+] Router check passed.")


if __name__ == "__main__":
    main()