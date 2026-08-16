"""
jarvis_cli/hardware.py — `jarvis --hardware`

Lightweight, read-only hardware/performance diagnostic:

    CPU          model name + logical core count
    RAM          total physical memory
    GPU          dedicated GPU(s), or "none — CPU-only mode"
    Ollama       version + base URL reachability
    Model        the resolved Ollama model + its on-disk size

Everything is defensive: any probe failure prints "unknown" instead of
crashing, and NO hardware or configuration changes are ever made.
"""

import ctypes
import os
import platform
import shutil
import subprocess

from utils.logger import get_logger

logger = get_logger("hardware")


def _cpu_name() -> str:
    """A friendly CPU model name (Windows registry, then platform)."""
    if os.name == "nt":
        try:
            import winreg

            key_path = (
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, key_path
            ) as key:
                name = str(winreg.QueryValueEx(key, "ProcessorNameString")[0])
                if name.strip():
                    return name.strip()
        except Exception as e:
            logger.debug(f"CPU registry lookup failed: {e}")
    try:
        name = platform.processor()
        if name:
            return name
    except Exception:
        pass
    return "unknown"


def _ram_gb() -> float | None:
    """Total physical RAM in GB, or None when undetectable."""
    if os.name == "nt":
        try:

            class MEMORYSTATUSEX(ctypes.Structure):  # noqa: N801
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
        except Exception as e:
            logger.debug(f"RAM detection failed (ctypes): {e}")
    try:
        import psutil  # optional dependency

        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        pass
    return None


def _gpu_names() -> list[str]:
    """Detect dedicated GPU(s): nvidia-smi first, then the Windows
    video-controller query. Returns [] when none is found."""
    names: list[str] = []

    nvidia = shutil.which("nvidia-smi")
    if nvidia:
        try:
            out = subprocess.run(
                [nvidia, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
            )
            if out.returncode == 0:
                names += [
                    line.strip()
                    for line in out.stdout.strip().splitlines()
                    if line.strip()
                ]
        except Exception as e:
            logger.debug(f"nvidia-smi probe failed: {e}")

    if os.name == "nt" and not names:
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 "Get-CimInstance Win32_VideoController "
                 "| Select-Object -ExpandProperty Name"],
                capture_output=True, text=True, timeout=15,
            )
            if out.returncode == 0:
                names += [
                    line.strip()
                    for line in out.stdout.strip().splitlines()
                    if line.strip()
                ]
        except Exception as e:
            logger.debug(f"Win32_VideoController probe failed: {e}")

    return names


def _ollama_version(base_url: str) -> str | None:
    """Ollama server version string, or None when unreachable."""
    import requests

    try:
        r = requests.get(f"{base_url}/api/version", timeout=3)
        r.raise_for_status()
        return str(r.json().get("version", "unknown"))
    except Exception as e:
        logger.debug(f"Ollama version probe failed: {e}")
        return None


def _model_size_gb(base_url: str, model: str) -> float | None:
    """On-disk size of *model* in GB, or None when unknown."""
    import requests

    try:
        r = requests.get(f"{base_url}/api/tags", timeout=3)
        r.raise_for_status()
        entries = r.json().get("models", [])
        # Exact name first — a prefix match on "qwen3" would grab
        # qwen3:1.7b when the user asked about qwen3:8b.
        for entry in entries:
            if entry.get("name") == model:
                size = entry.get("size")
                if isinstance(size, (int, float)) and size > 0:
                    return size / (1024 ** 3)
                return None
        # Tolerant fallback: the user's bare name may be reported
        # without a tag (e.g. pulled as just "qwen3").
        bare = model.split(":")[0]
        for entry in entries:
            if entry.get("name") == bare:
                size = entry.get("size")
                if isinstance(size, (int, float)) and size > 0:
                    return size / (1024 ** 3)
        return None
    except Exception as e:
        logger.debug(f"Model size probe failed: {e}")
        return None


def run_hardware() -> int:
    """Print the hardware report. Always returns 0 (read-only report)."""
    from config import jarvis_config, ollama_config

    print("\n=============================================")
    print("          JARVIS HARDWARE REPORT")
    print("=============================================")

    cpu = _cpu_name()
    cores = os.cpu_count() or "?"
    print(f"  CPU            {cpu} ({cores} logical cores)")

    ram = _ram_gb()
    print(f"  RAM            {ram:.1f} GB total" if ram else "  RAM            unknown")

    gpus = _gpu_names()
    if gpus:
        print("  GPU            " + ", ".join(gpus))
    else:
        print("  GPU            none detected - CPU-only mode")

    version = _ollama_version(ollama_config.BASE_URL)
    if version:
        print(f"  Ollama         {version} @ {ollama_config.BASE_URL}")
    else:
        print(f"  Ollama         not reachable @ {ollama_config.BASE_URL}")

    model = ollama_config.resolve_model()
    size = _model_size_gb(ollama_config.BASE_URL, model)
    size_txt = f"{size:.1f} GB" if size else "unknown"
    print(f"  Model          {model} ({size_txt})")
    print(
        f"  Model mode     {jarvis_config.MODEL_MODE or 'quality'} "
        f"(fast={ollama_config.FAST_MODEL or 'OLLAMA_MODEL'}, "
        f"quality={ollama_config.QUALITY_MODEL or 'OLLAMA_MODEL'})"
    )

    print("---------------------------------------------")
    # Only a dedicated NVIDIA (CUDA) GPU is usable by Ollama here.
    # Intel/Radeon integrated graphics are listed by Windows but cannot
    # accelerate local models, so the machine is effectively CPU-only.
    if any("nvidia" in g.lower() for g in gpus):
        print("  GPU acceleration available. Set OLLAMA_NUM_GPU=99 (the "
              "default) so Ollama offloads layers to it.")
    elif gpus:
        print("  CPU-only mode detected (integrated graphics found, but no "
              "dedicated NVIDIA GPU for Ollama). Smaller models and "
              "JARVIS_MODEL_MODE=fast give the lowest latency.")
    else:
        print("  CPU-only mode detected. Model generation runs on the "
              "processor - smaller models and JARVIS_MODEL_MODE=fast "
              "give the lowest latency.")
    print("  Read-only report - no hardware or configuration changes "
          "were made.")
    print("=============================================\n")
    return 0
