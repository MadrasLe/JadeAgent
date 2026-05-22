"""
Security primitives for mesh transport.

Includes:
- HMAC signing/verification for message integrity and authentication.
- Replay protection with timestamp + message-id tracking.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from dataclasses import dataclass
from typing import Any


def _canonical_json(payload: dict[str, Any]) -> str:
    """Canonical JSON representation for stable HMAC signatures."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


class HMACSigner:
    """Sign and verify payloads with HMAC-SHA256."""

    def __init__(self, secret: str, key_id: str = "mesh_v1"):
        if not secret:
            raise ValueError("HMAC secret cannot be empty.")
        self._secret = secret.encode("utf-8")
        self.key_id = key_id

    def sign(self, payload: dict[str, Any]) -> str:
        msg = _canonical_json(payload).encode("utf-8")
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def verify(self, payload: dict[str, Any], signature: str) -> bool:
        expected = self.sign(payload)
        return hmac.compare_digest(expected, signature)


@dataclass
class ReplayConfig:
    """Replay protection settings."""

    max_age_seconds: float = 120.0
    max_skew_seconds: float = 15.0
    max_entries: int = 20000


class ReplayProtector:
    """
    Reject repeated message ids and stale/future timestamps.

    Replay key format: "{source}:{message_id}".
    """

    def __init__(self, config: ReplayConfig | None = None):
        self.config = config or ReplayConfig()
        self._seen: dict[str, float] = {}

    def check(self, source: str, message_id: str, created_at: float) -> tuple[bool, str]:
        now = time.time()

        if created_at < now - self.config.max_age_seconds:
            return False, "Message too old."
        if created_at > now + self.config.max_skew_seconds:
            return False, "Message timestamp is too far in the future."

        key = f"{source}:{message_id}"
        previous = self._seen.get(key)
        if previous is not None:
            return False, "Replay detected for message id."

        self._seen[key] = now
        self._prune(now)
        return True, "ok"

    def _prune(self, now: float):
        expire_before = now - self.config.max_age_seconds
        stale = [k for k, ts in self._seen.items() if ts < expire_before]
        for key in stale:
            del self._seen[key]

        if len(self._seen) <= self.config.max_entries:
            return

        # If still too large, remove oldest entries.
        overshoot = len(self._seen) - self.config.max_entries
        oldest_keys = sorted(self._seen.items(), key=lambda item: item[1])[:overshoot]
        for key, _ in oldest_keys:
            del self._seen[key]

