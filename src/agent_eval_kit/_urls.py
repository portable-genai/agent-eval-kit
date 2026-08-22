"""The one transport rule both HTTP clients in this package obey.

Two clients here talk to a service over the network: the promotion-gate client and the
opt-in local-model judge. Both send evidence about a model's behaviour, so both refuse a
plaintext base URL unless it is loopback, where there is no network to eavesdrop on.

It lives in its own stdlib module rather than in either client because a security rule
copied into two files is two rules, and only one of them gets fixed when a defect is found.
This module imports nothing beyond the standard library, so importing it never pulls an HTTP
client into a consumer's decision core.
"""

from __future__ import annotations

from urllib.parse import urlparse

#: Hosts where plain ``http`` is acceptable, because the traffic never leaves the machine.
LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def require_secure_url(url: str, *, service: str, kind: str = "base") -> str:
    """Return ``url`` trimmed of a trailing slash; reject plaintext non-loopback URLs."""
    cleaned = (url or "").rstrip("/")
    parsed = urlparse(cleaned)
    if parsed.scheme == "https":
        return cleaned
    if parsed.scheme == "http" and (parsed.hostname or "") in LOOPBACK_HOSTS:
        return cleaned
    raise ValueError(
        f"{service}: refusing {kind} URL {url!r}: calls carrying evaluation evidence must use "
        "https:// (plain http is allowed only for localhost development)"
    )


__all__ = ["LOOPBACK_HOSTS", "require_secure_url"]
