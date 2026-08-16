"""
brain/memory.py — Conversation memory for JARVIS
 
[FIX m8] Added validation on load: role must be "user"|"assistant",
         content must be str ≤10000 chars. Max file size 1MB.
[FIX m11] Added disk-full handling with consecutive failure tracking.
[FIX m5] Added __all__ exports.
[FIX m1] Removed try/except config fallbacks.
"""
 
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import List, Dict
 
from config import jarvis_config, memory_config
from utils.logger import get_logger
 
__all__ = [
    "ConversationMemory",
    "Message",
]
 
logger = get_logger("memory")
 
Message = Dict[str, str]
 
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
 
# [FIX m8] Validation constants
_MAX_CONTENT_CHARS = 10000
_MAX_FILE_SIZE = 1024 * 1024  # 1MB
_VALID_ROLES = {"user", "assistant"}
 
 
def _resolve_path(cfg_file: str) -> str:
    p = Path(cfg_file)
    if p.is_absolute():
        return str(p)
    return str(_PROJECT_ROOT / p)
 
 
def _validate_message(msg: dict) -> bool:
    """[FIX m8] Validate a single message dict."""
    if not isinstance(msg, dict):
        return False
    
    role = msg.get("role")
    content = msg.get("content")
    
    if role not in _VALID_ROLES:
        return False
    
    if not isinstance(content, str):
        return False
    
    if len(content) > _MAX_CONTENT_CHARS:
        return False
    
    return True
 
 
class ConversationMemory:
    """Stores conversation history for context."""
 
    def __init__(
        self,
        max_turns: int | None = None,
        max_chars: int | None = None,
        persist_path: str | None = None,
    ):
        cfg_turns = jarvis_config.MEMORY_MAX_TURNS
        cfg_chars = jarvis_config.MEMORY_MAX_CHARS
        cfg_persist = memory_config.PERSIST
        cfg_file = memory_config.FILE
 
        self.max_turns = max_turns if max_turns is not None else cfg_turns
        self.max_chars = max_chars if max_chars is not None else cfg_chars
 
        if persist_path is not None:
            self._persist_path: str | None = persist_path or None
        elif cfg_persist and cfg_file:
            self._persist_path = _resolve_path(cfg_file)
        else:
            self._persist_path = None
 
        self._lock = threading.RLock()
        self._messages: List[Message] = []
        self._consecutive_save_failures = 0  # [FIX m11]
        self._load()
        logger.info(
            f"Memory initialized (max {self.max_turns} turns, "
            f"{self.max_chars} chars, persist={self._persist_path or 'off'})."
        )
 
    def add_user_message(self, content: str) -> None:
        if not content or not content.strip():
            return
        with self._lock:
            self._messages.append({
                "role": "user",
                "content": content.strip()[:_MAX_CONTENT_CHARS]  # [FIX m8] Truncate
            })
            self._enforce_limit()
        self._save()
 
    def add_assistant_message(self, content: str) -> None:
        if not content or not content.strip():
            return
        with self._lock:
            self._messages.append({
                "role": "assistant",
                "content": content.strip()[:_MAX_CONTENT_CHARS]  # [FIX m8] Truncate
            })
            self._enforce_limit()
        self._save()
 
    def _enforce_limit(self) -> None:
        """Drop oldest turns to stay within limits."""
        while len(self._messages) > self.max_turns * 2:
            self._messages.pop(0)
 
    def pop_last(self) -> Message | None:
        with self._lock:
            if not self._messages:
                return None
            return self._messages.pop()
 
    def get_history(self) -> List[Message]:
        with self._lock:
            return list(self._messages)

    def __len__(self) -> int:
        with self._lock:
            return len(self._messages)
 
    def get_context_for_ollama(self, system_prompt: str) -> List[Message]:
        return self.get_trimmed_context(system_prompt)
 
    def get_trimmed_context(self, system_prompt: str) -> List[Message]:
        with self._lock:
            history = list(self._messages)
 
        messages: List[Message] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(history)
 
        if self.max_chars > 0:
            total = sum(len(m["content"]) for m in messages)
            while total > self.max_chars and len(messages) > 3:
                removed = messages.pop(1)
                total -= len(removed["content"])
                if len(messages) > 2 and messages[1]["role"] == "assistant":
                    removed = messages.pop(1)
                    total -= len(removed["content"])
 
        return messages
 
    def clear(self) -> None:
        with self._lock:
            self._messages.clear()
        self._save()
        logger.info("Conversation memory cleared.")
 
    def _save(self) -> None:
        """[FIX m11] Write history with disk-full handling."""
        if not self._persist_path:
            return
        
        try:
            path = Path(self._persist_path)
            
            # [FIX m11] Check available disk space
            try:
                usage = shutil.disk_usage(path.parent if path.parent.exists() else Path.home())
                if usage.free < 10 * 1024 * 1024:  # Less than 10MB free
                    self._consecutive_save_failures += 1
                    if self._consecutive_save_failures >= 3:
                        logger.error("Low disk space! Memory persistence may fail.")
                    return
            except Exception:
                pass  # disk_usage may fail on some systems
            
            path.parent.mkdir(parents=True, exist_ok=True)
            
            fd, tmp = tempfile.mkstemp(
                prefix=".conversation_", suffix=".tmp",
                dir=str(path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._messages, f, ensure_ascii=False, indent=2)
                os.replace(tmp, str(path))
                self._consecutive_save_failures = 0  # [FIX m11] Reset on success
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            self._consecutive_save_failures += 1  # [FIX m11]
            if self._consecutive_save_failures >= 3:
                logger.error(f"Memory save failed {self._consecutive_save_failures} times: {e}")
            else:
                logger.warning(f"Could not persist conversation memory: {e}")
 
    def _load(self) -> None:
        """[FIX m8] Load with validation."""
        if not self._persist_path:
            return
        
        path = Path(self._persist_path)
        if not path.is_file():
            return
        
        try:
            # [FIX m8] Check file size
            file_size = path.stat().st_size
            if file_size > _MAX_FILE_SIZE:
                logger.warning(
                    f"Conversation file too large ({file_size} bytes). "
                    "Starting fresh."
                )
                self._backup_and_reset(path)
                return
            
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if not isinstance(data, list):
                logger.warning("Conversation file corrupt (not a list). Starting fresh.")
                self._backup_and_reset(path)
                return
            
            # [FIX m8] Validate each message
            valid_messages = []
            for msg in data:
                if _validate_message(msg):
                    valid_messages.append({
                        "role": msg["role"],
                        "content": msg["content"][:_MAX_CONTENT_CHARS]
                    })
                else:
                    logger.warning("Skipping invalid message in conversation file")
            
            self._messages = valid_messages
            logger.info(f"Loaded {len(self._messages)} messages from {path}")
            
        except json.JSONDecodeError as e:
            logger.warning(f"Conversation file corrupt: {e}. Starting fresh.")
            self._backup_and_reset(path)
        except Exception as e:
            logger.warning(f"Could not load conversation memory: {e}")
 
    def _backup_and_reset(self, path: Path) -> None:
        """Backup corrupt file and start fresh."""
        try:
            backup = path.with_name(
                f"{path.stem}.corrupt-{int(time.time())}{path.suffix}"
            )
            os.replace(str(path), str(backup))
            logger.info(f"Backed up corrupt file to {backup}")
        except Exception:
            pass
        self._messages = []
        self._save()