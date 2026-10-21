"""Opt-in full-content pagination, wired into all four tools.

Page 1 of every tool stays trimmed (Task 2) and, when anything was trimmed,
carries a top-level `truncation` marker plus an opaque `cursor` (Task 3). A
client that wants more calls the same tool again with that `cursor`; the tool
re-fetches the feed untrimmed, concatenates entry text in return order, and
serves the next `_page_size()`-sized slice via `_expand`.

The first expansion call confirms once via `ctx.elicit` (Task 1 POC: this
needs a stateful transport, wired up in a later deploy task — not exercised
here, so we fake `ctx`). Accept serves the slice with a now-confirmed next
cursor; decline serves nothing; a client that doesn't support `elicit` (or
any other transport failure) falls back to serving the slice anyway, flagged
`elicited: False`, rather than hanging or erroring.

No network: `_fetch_feed` is monkeypatched to run the real `_refine_entry`
over canned Atom XML, so entries carry the exact shape the real producer
emits instead of a hand-crafted dict fixture.
"""

from __future__ import annotations

import asyncio
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import pytest

import server


def _atom_entry(content: str, kind: str = "t3") -> ET.Element:
    xml_text = (
        '<entry xmlns="http://www.w3.org/2005/Atom">'
        f"<id>{kind}_abc123</id>"
        "<title>A post</title>"
        f"<content>{content}</content>"
        "</entry>"
    )
    return ET.fromstring(xml_text)


def _fake_fetch_feed(entries_xml: list[ET.Element], feed_title: str = "feed"):
    """Build a `_fetch_feed` replacement that runs the real `_refine_entry`
    over canned Atom XML, honoring the caller's `limit` instead of hitting
    the network — so the entry shape stays anchored to the real producer.
    """

    async def _fetch(url, params=None, limit=server.TEXT_MAX):
        refined = [server._refine_entry(e, limit) for e in entries_xml]
        return feed_title, refined

    return _fetch


class FakeContext:
    """Stands in for `Context.elicit`.

    `action` mimics `ElicitationResult.action` ("accept" / "decline" /
    "cancel"); `raise_error=True` mimics a client that doesn't support
    elicitation (or any other transport failure).
    """

    def __init__(self, action: str = "accept", raise_error: bool = False):
        self.action = action
        self.raise_error = raise_error
        self.elicit_calls = 0

    async def elicit(self, message: str, schema):
        self.elicit_calls += 1
        if self.raise_error:
            raise RuntimeError("elicit not supported by this client")
        return SimpleNamespace(action=self.action, data=None)


@pytest.fixture(autouse=True)
def _default_page_size(monkeypatch):
    # Pin page size to TEXT_MAX for every test in this module, regardless of
    # the operator's shell environment.
    monkeypatch.delenv("REDDIT_MCP_PAGE_SIZE", raising=False)


def test_default_call_sets_truncation_marker_and_cursor(monkeypatch):
    long_text = "z" * 5000
    monkeypatch.setattr(server, "_fetch_feed", _fake_fetch_feed([_atom_entry(long_text)]))

    result = asyncio.run(server.browse_subreddit("python"))

    assert result["truncation"]["truncated"] is True
    assert result["truncation"]["entries_trimmed"] == 1
    assert result["truncation"]["total"] == len(long_text)
    assert "elicited" not in result["truncation"]
    assert result.get("cursor")
    offset, confirmed = server._decode_cursor(result["cursor"])
    assert (offset, confirmed) == (server._page_size(), False)


def test_expansion_elicit_accept_returns_next_slice(monkeypatch):
    full = "z" * 5000
    monkeypatch.setattr(server, "_fetch_feed", _fake_fetch_feed([_atom_entry(full)]))
    page_size = server._page_size()
    cursor = server._encode_cursor(page_size, False)
    ctx = FakeContext(action="accept")

    result = asyncio.run(server.browse_subreddit("python", cursor=cursor, ctx=ctx))

    assert result["page_text"] == full[page_size : page_size * 2]
    assert ctx.elicit_calls == 1
    offset, confirmed = server._decode_cursor(result["cursor"])
    assert offset == page_size * 2
    assert confirmed is True


def test_expansion_elicit_decline_returns_empty(monkeypatch):
    full = "z" * 5000
    monkeypatch.setattr(server, "_fetch_feed", _fake_fetch_feed([_atom_entry(full)]))
    cursor = server._encode_cursor(server._page_size(), False)
    ctx = FakeContext(action="decline")

    result = asyncio.run(server.browse_subreddit("python", cursor=cursor, ctx=ctx))

    assert result["page_text"] == ""
    assert result["truncation"]["declined"] is True
    assert result["truncation"]["elicited"] is True


def test_confirmed_cursor_skips_elicit(monkeypatch):
    full = "z" * 5000
    monkeypatch.setattr(server, "_fetch_feed", _fake_fetch_feed([_atom_entry(full)]))
    page_size = server._page_size()
    cursor = server._encode_cursor(page_size, True)
    ctx = FakeContext(action="accept")

    result = asyncio.run(server.browse_subreddit("python", cursor=cursor, ctx=ctx))

    assert ctx.elicit_calls == 0
    assert result["page_text"] == full[page_size : page_size * 2]


def test_elicit_unsupported_falls_back(monkeypatch):
    full = "z" * 5000
    monkeypatch.setattr(server, "_fetch_feed", _fake_fetch_feed([_atom_entry(full)]))
    page_size = server._page_size()
    cursor = server._encode_cursor(page_size, False)
    ctx = FakeContext(raise_error=True)

    result = asyncio.run(server.browse_subreddit("python", cursor=cursor, ctx=ctx))

    assert result["truncation"]["elicited"] is False
    assert result["page_text"] == full[page_size : page_size * 2]


def test_offset_past_end_returns_empty_not_error(monkeypatch):
    full = "z" * 100
    monkeypatch.setattr(server, "_fetch_feed", _fake_fetch_feed([_atom_entry(full)]))
    cursor = server._encode_cursor(10_000, True)
    ctx = FakeContext(action="accept")

    result = asyncio.run(server.browse_subreddit("python", cursor=cursor, ctx=ctx))

    assert result["page_text"] == ""
    assert result["truncation"]["truncated"] is False
    assert result.get("cursor") is None
    assert ctx.elicit_calls == 0


def test_fetch_json_nonjson_uses_marker(monkeypatch):
    """The allow-any-url, non-JSON-response fallback reports structured trim
    fields (matching `_refine_entry`'s shape) instead of an inline
    "[truncated ...]" suffix baked into the text.
    """

    class _NonJSONResponse:
        def __init__(self, text):
            self.status_code = 200
            self.headers = {"content-type": "text/plain"}
            self.url = "http://example.com/data"
            self.text = text

        @property
        def is_success(self):
            return True

        def json(self):
            raise ValueError("not json")

        def raise_for_status(self):
            return None

    class _NonJSONClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def get(self, url, params=None):
            return _NonJSONResponse("x" * 6000)

    monkeypatch.setenv("REDDIT_MCP_ALLOW_ANY_URL", "1")
    monkeypatch.setattr(server.httpx, "AsyncClient", _NonJSONClient)

    result = asyncio.run(server.fetch_json("https://example.com/data"))

    assert result["text_total_chars"] == 6000
    assert result["text_truncated"] is True
    assert len(result["text"]) == 5000
    assert "[truncated" not in (result["text"] or "")


def test_refine_entry_full_text_path_feeds_expansion():
    """Deferred Task 2 note: `_refine_entry(limit=None)` must return the
    full, untrimmed text — the cursor-path refetch in every tool relies on
    exactly this to rebuild `full_text` for `_expand`.
    """
    full_text = "q" * 9000
    refined = server._refine_entry(_atom_entry(full_text), limit=None)
    assert refined["text"] == full_text
    assert refined["text_total_chars"] == len(full_text)
    assert refined["text_truncated"] is False
