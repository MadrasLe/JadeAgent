"""LLM Backend providers."""

from .base import LLMBackend
from .openai_compat import OpenAICompatBackend

__all__ = ["LLMBackend", "OpenAICompatBackend"]

# Lazy import for MegaGemm (requires GPU)
def MegaGemmBackend(*args, **kwargs):
    from .megagemm import MegaGemmBackend as _MGB
    return _MGB(*args, **kwargs)
