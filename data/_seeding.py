"""Stable seed derivation for the demo providers.

`zlib.crc32` of `prefix + key` is stable across Python processes,
unlike `hash()` which is salted by PYTHONHASHSEED. The demo providers
promise 'deterministic per symbol'; using crc32 keeps that promise.
"""
from __future__ import annotations

import zlib


def deterministic_seed(prefix: str, key: str) -> int:
    """A 32-bit seed derived from `prefix + key`, stable across processes."""
    return zlib.crc32(f"{prefix}{key}".encode("utf-8"))
