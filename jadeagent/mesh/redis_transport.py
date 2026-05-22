"""
Redis-backed mesh transport with optional TLS + HMAC authentication.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .protocol import MeshEnvelope, envelope_from_dict, envelope_to_dict
from .security import HMACSigner, ReplayProtector

logger = logging.getLogger("jadeagent.mesh.redis_transport")


class RedisMeshTransport:
    """
    Publish/subscribe transport for distributed mesh nodes.

    Security features:
    - TLS for encrypted transport (set `tls=True` and Redis configured for SSL).
    - Optional HMAC signature verification (`signer`).
    - Optional replay protection (`replay_protector`).
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        channel_prefix: str = "jade:mesh:",
        tls: bool = False,
        tls_ca_certs: str | None = None,
        tls_certfile: str | None = None,
        tls_keyfile: str | None = None,
        tls_cert_reqs: str | None = "required",
        signer: HMACSigner | None = None,
        replay_protector: ReplayProtector | None = None,
        poll_timeout: float = 0.01,
        redis_kwargs: dict[str, Any] | None = None,
    ):
        # Initialize defensively so close/__del__ are safe on partial init.
        self._pubsubs: dict[str, Any] = {}
        self._client: Any | None = None

        try:
            import redis
        except ImportError as exc:
            raise ImportError(
                "RedisMeshTransport requires redis package. Install with: pip install redis"
            ) from exc

        self._redis = redis
        self.redis_url = redis_url
        self.channel_prefix = channel_prefix
        self.signer = signer
        self.replay = replay_protector
        self.poll_timeout = poll_timeout

        kwargs = dict(redis_kwargs or {})
        if tls:
            kwargs.setdefault("ssl", True)
            if tls_ca_certs:
                kwargs["ssl_ca_certs"] = tls_ca_certs
            if tls_certfile:
                kwargs["ssl_certfile"] = tls_certfile
            if tls_keyfile:
                kwargs["ssl_keyfile"] = tls_keyfile
            if tls_cert_reqs:
                kwargs["ssl_cert_reqs"] = tls_cert_reqs

        self._client = redis.Redis.from_url(redis_url, decode_responses=False, **kwargs)
        self._client.ping()

    @property
    def broadcast_channel(self) -> str:
        return f"{self.channel_prefix}broadcast"

    def _node_channel(self, node_id: str) -> str:
        return f"{self.channel_prefix}node:{node_id}"

    def register(self, node: Any):
        node_id = getattr(node, "node_id", None) or str(node)
        self._ensure_subscription(node_id)

    def unregister(self, node_id: str):
        pubsub = self._pubsubs.pop(node_id, None)
        if pubsub is None:
            return
        try:
            pubsub.close()
        except Exception:
            pass

    def send(self, envelope: MeshEnvelope) -> int:
        envelope_data = envelope_to_dict(envelope)
        wire_payload = self._encode_wire(envelope_data)

        if envelope.destination is None:
            channel = self.broadcast_channel
        else:
            channel = self._node_channel(envelope.destination)

        delivered = self._client.publish(channel, wire_payload)
        return int(delivered)

    def poll(self, node_id: str, max_messages: int = 32) -> list[MeshEnvelope]:
        pubsub = self._ensure_subscription(node_id)
        envelopes: list[MeshEnvelope] = []

        for _ in range(max_messages):
            item = pubsub.get_message(timeout=self.poll_timeout)
            if item is None:
                break
            if item.get("type") != "message":
                continue

            raw_data = item.get("data")
            channel = item.get("channel")
            envelope = self._decode_wire(raw_data, channel)
            if envelope is None:
                continue
            envelopes.append(envelope)

        return envelopes

    def close(self):
        for node_id in list(self._pubsubs.keys()):
            self.unregister(node_id)
        client = self._client
        if client is None:
            return
        try:
            client.close()
        except Exception:
            pass

    def _ensure_subscription(self, node_id: str):
        existing = self._pubsubs.get(node_id)
        if existing is not None:
            return existing

        pubsub = self._client.pubsub(ignore_subscribe_messages=True)
        pubsub.subscribe(self._node_channel(node_id), self.broadcast_channel)
        self._pubsubs[node_id] = pubsub
        return pubsub

    def _encode_wire(self, envelope_data: dict[str, Any]) -> bytes:
        wire = {"envelope": envelope_data}
        if self.signer is not None:
            wire["auth"] = {
                "scheme": "hmac-sha256",
                "key_id": self.signer.key_id,
                "signature": self.signer.sign(envelope_data),
            }
        return json.dumps(wire, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

    def _decode_wire(self, raw_data: Any, channel: Any) -> MeshEnvelope | None:
        try:
            if isinstance(raw_data, bytes):
                raw_text = raw_data.decode("utf-8")
            else:
                raw_text = str(raw_data)
            wire = json.loads(raw_text)
        except Exception:
            logger.warning("Dropping invalid JSON wire message.")
            return None

        envelope_data = wire.get("envelope")
        if not isinstance(envelope_data, dict):
            logger.warning("Dropping wire message without envelope payload.")
            return None

        if self.signer is not None:
            auth = wire.get("auth", {})
            signature = auth.get("signature") if isinstance(auth, dict) else None
            if not isinstance(signature, str):
                logger.warning("Dropping unsigned wire message.")
                return None
            if not self.signer.verify(envelope_data, signature):
                logger.warning("Dropping wire message with invalid signature.")
                return None

        source = str(envelope_data.get("source", ""))
        message_id = str(envelope_data.get("message_id", ""))
        created_at = float(envelope_data.get("created_at", 0.0))
        if self.replay is not None:
            ok, reason = self.replay.check(source, message_id, created_at)
            if not ok:
                logger.warning("Dropping replay/stale message: %s", reason)
                return None

        try:
            return envelope_from_dict(envelope_data)
        except Exception as exc:
            logger.warning("Failed to decode envelope: %s", exc)
            return None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
