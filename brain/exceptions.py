"""
brain/exceptions.py — Custom exceptions for JARVIS brain module.
"""
 
__all__ = [
    "JARVISError",
    "OllamaTimeoutError",
    "ProviderUnavailableError",
    "CircuitOpenError",
]
 
 
class JARVISError(Exception):
    """Base exception for JARVIS errors."""
    pass
 
 
class OllamaTimeoutError(JARVISError):
    """Raised when Ollama request times out."""
    pass
 
 
class ProviderUnavailableError(JARVISError):
    """Raised when the AI provider is unavailable."""
    pass
 
 
class CircuitOpenError(JARVISError):
    """Raised when circuit breaker is open."""
    pass
 