"""Opaque pagination cursor: encode/decode an (offset, confirmed) position.

`_encode_cursor` / `_decode_cursor` let a later pagination tool hand clients
an opaque token instead of a raw offset. `_decode_cursor` must never raise —
a missing, forged, or stale cursor should fall back to (0, False) (start
over, unconfirmed) rather than error out the tool call.
"""

from __future__ import annotations

import server


def test_roundtrip():
    assert server._decode_cursor(server._encode_cursor(2000, True)) == (2000, True)


def test_decode_garbage_is_safe():
    assert server._decode_cursor("not-a-cursor") == (0, False)


def test_decode_none():
    assert server._decode_cursor(None) == (0, False)


def test_decode_negative_offset_clamped():
    token = server._encode_cursor(-5, True)
    offset, _ = server._decode_cursor(token)
    assert offset == 0
