"""Lightweight logging and timing helpers."""

import time
from typing import Any


def info(msg: Any) -> None:
    print(f"[i] {msg}")


def status(msg: Any) -> None:
    print(f"[*] {msg}")


def ok(msg: Any) -> None:
    print(f"[+] {msg}")


def warning(msg: Any) -> None:
    print(f"[!] {msg}")


def error(msg: Any) -> None:
    print(f"[x] {msg}")


def tick() -> float:
    return time.perf_counter()


def report(label: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    print(f"[{label}] {elapsed:.2f} sec")
    return elapsed