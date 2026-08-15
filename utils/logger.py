"""Lightweight logging and timing helpers."""

import time


def info(msg):
    print(f"[i] {msg}")


def status(msg):
    print(f"[*] {msg}")


def ok(msg):
    print(f"[+] {msg}")


def warning(msg):
    print(f"[!] {msg}")


def error(msg):
    print(f"[x] {msg}")


def tick():
    return time.perf_counter()


def report(label, start):
    elapsed = time.perf_counter() - start
    print(f"[{label}] {elapsed:.2f} sec")
    return elapsed