#!/usr/bin/env python3
"""Local MCP server that reads Reddit through its public RSS feeds.

Reddit gates its Data API (new apps require a moderation use case + approval)
and blocks anonymous ``.json`` access, but it still publishes public **Atom/RSS
feeds** for reading content. These tools fetch those feeds and trim them to the
useful fields. No authentication, account, or app registration required.

Feeds carry title, author, link, timestamp, and full post/comment text — but
NOT scores, upvote ratios, or comment counts (those only exist in the gated API).

``fetch_json`` covers the remaining Reddit feed shapes (user, multireddit,
domain). It refuses other hosts unless REDDIT_MCP_ALLOW_ANY_URL=1 is set.
"""

from __future__ import annotations

import asyncio
import base64
import html
import json
import os
import random
import re
import time
import xml.etree.ElementTree as ET
from urllib.parse import urlsplit, urlunsplit

import httpx
from mcp.server.mcpserver import Context, MCPServer
from pydantic import BaseModel

# Feed readers send their own User-Agent; Reddit serves RSS to them normally.
USER_AGENT = os.environ.get("REDDIT_USER_AGENT", "reddit-mcp/1.0 (personal RSS reader)")
REDDIT_BASE = "https://www.reddit.com"
ATOM = {"a": "http://www.w3.org/2005/Atom"}
TIMEOUT = 30.0
TEXT_MAX = 2000

# Reddit throttles its unauthenticated RSS feeds aggressively. Retry 429s with
# backoff (honoring Retry-After) and keep a global minimum gap between requests
# rather than failing fast.
MAX_RETRIES = 7  # retries after the first attempt: ~2,4,8,16,32,64,128s backoff
BACKOFF_BASE = 2.0  # seconds; per-retry sleep is BACKOFF_BASE * 2**attempt + jitter
MIN_INTERVAL = 1.0  # minimum seconds between outbound requests

mcp = MCPServer("reddit")

# Serialize the throttle so concurrent tool calls still respect MIN_INTERVAL.
_throttle_lock = asyncio.Lock()
_last_request_at = 0.0


async def _throttle() -> None:
    """Sleep so successive outbound requests are at least MIN_INTERVAL apart."""
    global _last_request_at
    if MIN_INTERVAL <= 0:
        return
    async with _throttle_lock:
        wait = MIN_INTERVAL - (time.monotonic() - _last_request_at)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_request_at = time.monotonic()


def _parse_retry_after(value: str | None) -> float | None:
    """Reddit sends Retry-After as integer seconds; ignore the HTTP-date form."""
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None


async def _get_with_retry(
    client: httpx.AsyncClient, url: str, params: dict | None = None
) -> httpx.Response:
    """GET with throttling and 429 retry/backoff.

    On a 429 we wait for the Retry-After header (if present) or an exponential
    backoff with jitter, retrying up to MAX_RETRIES times. Only after exhausting
    retries do we raise the rate-limited error, noting attempts and the last
    Retry-After seen.
    """
    last_retry_after: float | None = None
    for attempt in range(MAX_RETRIES + 1):
        await _throttle()
        resp = await client.get(url, params=params)
        if resp.status_code != 429:
            return resp
        last_retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
        if attempt == MAX_RETRIES:
            break
        if last_retry_after is not None:
            delay = last_retry_after
        else:
            delay = BACKOFF_BASE * (2**attempt) + random.random()
        await asyncio.sleep(delay)

    attempts = MAX_RETRIES + 1
    suffix = (
        f"; last Retry-After was {last_retry_after:g}s"
        if last_retry_after is not None
        else ""
    )
    raise RuntimeError(
        f"Reddit rate-limited this request (HTTP 429) after {attempts} "
        f"attempt(s){suffix}. Wait a bit and retry."
    )


# --------------------------------------------------------------------------- #
# Fetch + parse
# --------------------------------------------------------------------------- #
async def _fetch_feed(
    url: str, params: dict | None = None, limit: int | None = TEXT_MAX
) -> tuple[str | None, list[dict]]:
    """GET an Atom feed and return (feed_title, [refined entries]).

    `limit` is forwarded to `_refine_entry` as its per-entry text trim limit;
    pass `limit=None` to fetch full, untrimmed entry text (used by the
    cursor-expansion path below, which re-fetches the same feed to rebuild
    the full text it paginates over).
    """
    headers = {"User-Agent": USER_AGENT}
    async with httpx.AsyncClient(
        timeout=TIMEOUT, follow_redirects=True, headers=headers
    ) as client:
        resp = await _get_with_retry(client, url, params)
    resp.raise_for_status()
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError as e:
        raise RuntimeError(f"Could not parse feed (not valid Atom XML): {e}")
    feed_title = root.findtext("a:title", namespaces=ATOM)
    entries = [_refine_entry(e, limit) for e in root.findall("a:entry", ATOM)]
    return feed_title, entries


# --------------------------------------------------------------------------- #
# Refinement helpers
# --------------------------------------------------------------------------- #
def _trim_field(text: str | None, limit: int | None) -> tuple[str | None, int, bool]:
    """Trim text to `limit` chars, reporting the full length and whether trimmed.

    Returns (shown, total_chars, truncated). `limit=None` returns the full
    text untrimmed.
    """
    if not text:
        return text, 0, False
    total = len(text)
    if limit is None or total <= limit:
        return text, total, False
    return text[:limit], total, True


def _html_to_text(raw: str | None, limit: int | None = TEXT_MAX) -> tuple[str | None, int, bool]:
    """Convert the HTML in a feed <content> to readable plain text.

    Returns (shown, total_chars, truncated) via `_trim_field`.
    """
    if not raw:
        return raw, 0, False
    s = re.sub(r"(?is)<(script|style).*?</\1>", "", raw)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p>", "\n\n", s)
    s = re.sub(r"(?s)<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"\n{3,}", "\n\n", s).strip()
    return _trim_field(s, limit)


def _refine_entry(entry: ET.Element, limit: int | None = TEXT_MAX) -> dict:
    """Reduce an Atom entry (post or comment) to the useful fields."""
    full_id = entry.findtext("a:id", default="", namespaces=ATOM)
    kind, _, short_id = full_id.partition("_")  # e.g. "t3_abc123"
    link_el = entry.find("a:link", ATOM)
    category = entry.find("a:category", ATOM)
    text, total_chars, truncated = _html_to_text(
        entry.findtext("a:content", namespaces=ATOM), limit
    )
    return {
        "id": short_id or full_id,
        "type": {"t3": "post", "t1": "comment"}.get(kind, kind or None),
        "title": entry.findtext("a:title", namespaces=ATOM),
        "author": entry.findtext("a:author/a:name", namespaces=ATOM),
        "subreddit": category.get("term") if category is not None else None,
        "published": entry.findtext("a:published", namespaces=ATOM),
        "updated": entry.findtext("a:updated", namespaces=ATOM),
        "link": link_el.get("href") if link_el is not None else None,
        "text": text,
        "text_total_chars": total_chars,
        "text_truncated": truncated,
    }


# --------------------------------------------------------------------------- #
# Pagination cursor helpers
#
# Opaque token wrapping an (offset, confirmed) position for a later paginated
# "full content" tool. Clients must treat it as opaque; `_decode_cursor` must
# never raise on a missing/forged/stale token — fall back to (0, False)
# (start over, unconfirmed) instead.
# --------------------------------------------------------------------------- #
def _encode_cursor(offset: int, confirmed: bool) -> str:
    """Encode a pagination position as an opaque, URL-safe cursor token."""
    payload = json.dumps({"o": offset, "c": confirmed}, separators=(",", ":"))
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")


def _decode_cursor(token: str | None) -> tuple[int, bool]:
    """Decode a cursor from `_encode_cursor`, tolerating any invalid input."""
    if not token:
        return 0, False
    try:
        data = json.loads(base64.urlsafe_b64decode(token.encode("ascii")))
        offset, confirmed = data["o"], data["c"]
        if isinstance(offset, bool) or not isinstance(offset, int) or not isinstance(confirmed, bool):
            return 0, False
    except Exception:
        return 0, False
    return max(offset, 0), confirmed


# --------------------------------------------------------------------------- #
# Opt-in full-content pagination
#
# Page 1 of every tool stays trimmed and, when anything was trimmed, carries
# a top-level `truncation` marker plus a `cursor`. A client that wants more
# calls the same tool again with that cursor; `_paginate` re-fetches the feed
# untrimmed and hands the concatenated text to `_expand`, which confirms once
# via `ctx.elicit` before serving each page (Task 1 POC: elicit needs a
# stateful transport — wired up in a later deploy task, not here).
# --------------------------------------------------------------------------- #
def _page_size() -> int:
    """Page size for cursor-paginated expansion: REDDIT_MCP_PAGE_SIZE, or TEXT_MAX."""
    raw = os.environ.get("REDDIT_MCP_PAGE_SIZE", "").strip()
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return TEXT_MAX


class _ConfirmFull(BaseModel):
    """Yes/no elicit schema: confirm before serving a page of full content."""

    confirm: bool


def _build_truncation_marker(entries: list[dict], page_size: int) -> tuple[dict, str | None]:
    """Top-level `truncation` marker + opt-in cursor for a page-1 response.

    `total` sums each entry's `text_total_chars` (the full, untrimmed length)
    so it lines up with what `_expand` computes once a client re-fetches the
    same entries with `limit=None` and concatenates them in return order.
    """
    entries_trimmed = sum(1 for e in entries if e.get("text_truncated"))
    truncated = entries_trimmed > 0
    total = sum(e.get("text_total_chars") or 0 for e in entries)
    marker = {"truncated": truncated, "total": total, "entries_trimmed": entries_trimmed}
    cursor = _encode_cursor(page_size, False) if truncated else None
    return marker, cursor


async def _expand(ctx: Context | None, full_text: str, token: str | None) -> dict:
    """Serve one cursor-paginated page of `full_text`.

    Returns {"page_text", "cursor", "truncation": {...}}. An unconfirmed
    cursor triggers one `ctx.elicit` confirmation: accept serves the slice
    with a now-confirmed next cursor; decline serves nothing
    (`truncation.declined`). If elicit raises (client doesn't support it, or
    any other transport error), fall back to serving the slice anyway with
    `truncation.elicited = False` instead of hanging or erroring.
    """
    offset, confirmed = _decode_cursor(token)
    total = len(full_text)
    page_size = _page_size()

    if offset >= total:
        return {
            "page_text": "",
            "cursor": None,
            "truncation": {"truncated": False, "shown": 0, "total": total, "elicited": True},
        }

    elicited = True
    if not confirmed:
        try:
            result = await ctx.elicit(
                message=(
                    f"Show the next {min(page_size, total - offset)} characters "
                    f"of {total} total?"
                ),
                schema=_ConfirmFull,
            )
        except Exception:
            # Client doesn't support elicitation, or some other transport
            # error — fall back to serving the page rather than hanging.
            elicited = False
        else:
            if result.action != "accept":
                return {
                    "page_text": "",
                    "cursor": None,
                    "truncation": {
                        "truncated": True,
                        "shown": 0,
                        "total": total,
                        "elicited": True,
                        "declined": True,
                    },
                }

    page_text = full_text[offset : offset + page_size]
    next_offset = offset + page_size
    # Once a page has been served (accepted, already confirmed, or served via
    # the unsupported-elicit fallback), treat the position as confirmed so a
    # client that doesn't support elicit isn't asked again on every page.
    next_cursor = _encode_cursor(next_offset, True) if next_offset < total else None
    return {
        "page_text": page_text,
        "cursor": next_cursor,
        "truncation": {
            "truncated": next_cursor is not None,
            "shown": len(page_text),
            "total": total,
            "elicited": elicited,
        },
    }


async def _paginate(ctx: Context | None, cursor: str | None, fetch_full) -> dict | None:
    """Shared cursor branch for all four tools.

    `fetch_full` is an async, no-arg callable that re-fetches the same feed
    with `limit=None` and returns entries in the same order used for the
    page-1 truncation marker. Returns None when `cursor` is absent, so the
    caller proceeds with its normal trimmed page-1 response.
    """
    if cursor is None:
        return None
    entries = await fetch_full()
    full_text = "".join(e.get("text") or "" for e in entries)
    return await _expand(ctx, full_text, cursor)


def _is_reddit_host(netloc: str) -> bool:
    host = netloc.lower().split(":")[0]
    return host == "reddit.com" or host.endswith(".reddit.com")


def _allow_any_url() -> bool:
    """Whether the operator has opted in to fetching non-Reddit hosts."""
    return os.environ.get("REDDIT_MCP_ALLOW_ANY_URL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
async def browse_subreddit(
    subreddit: str,
    sort: str = "hot",
    time_filter: str = "day",
    limit: int = 25,
    cursor: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Read what a subreddit is posting right now, to see a community's live activity.

    Use this when the question is about what people in a specific community are
    currently saying or doing, or when a post is too recent for Reddit's search
    index to have caught it yet (search lags; "new" here does not).

    Scores, upvote ratios, and comment counts are not available.

    Args:
        subreddit: Subreddit name, with or without the "r/" prefix.
        sort: One of hot, new, top, rising, controversial.
        time_filter: For top/controversial — hour, day, week, month, year, all.
        limit: Number of posts (1-100).
        cursor: Opaque token from a previous call's `cursor`, to page through
            a post's full, untrimmed text instead of the default trimmed page.
    """
    sub = subreddit.strip().removeprefix("r/").strip("/")
    sort = sort.strip().lower()
    limit = max(1, min(int(limit), 100))
    params: dict[str, str | int] = {"limit": limit}
    if sort in ("top", "controversial"):
        params["t"] = time_filter
    feed_url = f"{REDDIT_BASE}/r/{sub}/{sort}/.rss"

    async def _fetch_full():
        _, full_entries = await _fetch_feed(feed_url, params, limit=None)
        return full_entries

    expanded = await _paginate(ctx, cursor, _fetch_full)
    if expanded is not None:
        return expanded

    title, entries = await _fetch_feed(feed_url, params)
    marker, next_cursor = _build_truncation_marker(entries, _page_size())
    result = {
        "subreddit": sub,
        "sort": sort,
        "feed_title": title,
        "count": len(entries),
        "posts": entries,
        "truncation": marker,
    }
    if next_cursor is not None:
        result["cursor"] = next_cursor
    return result


@mcp.tool()
async def get_post(
    url: str,
    comment_limit: int = 50,
    cursor: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Read one Reddit thread in full: the original post plus its comments.

    Use this after a search or a browse turns up a promising thread and you need
    what people actually wrote in it, the details, caveats, corrections, and
    disagreements that a title alone never carries.

    Comments come back as a flat list (the reply tree is not available), and
    neither are scores or vote counts.

    Args:
        url: A Reddit post URL or permalink (e.g.
            https://www.reddit.com/r/python/comments/abc123/title/ or
            /r/python/comments/abc123/title/).
        comment_limit: Max comments to return.
        cursor: Opaque token from a previous call's `cursor`, to page through
            the post's and comments' full, untrimmed text.
    """
    u = url.strip()
    path = u if u.startswith("/") else urlsplit(u).path
    path = path.rstrip("/")
    if path.endswith("/.rss"):
        path = path[: -len("/.rss")]
    feed_url = f"{REDDIT_BASE}{path}/.rss"

    def _ordered(entries: list[dict]) -> tuple[list[dict], list[dict]]:
        posts = [e for e in entries if e["type"] == "post"]
        comments = [e for e in entries if e["type"] == "comment"][:comment_limit]
        return posts, comments

    async def _fetch_full():
        _, full_entries = await _fetch_feed(feed_url, {"limit": comment_limit}, limit=None)
        posts_f, comments_f = _ordered(full_entries)
        return ([posts_f[0]] if posts_f else []) + comments_f

    expanded = await _paginate(ctx, cursor, _fetch_full)
    if expanded is not None:
        return expanded

    _, entries = await _fetch_feed(feed_url, {"limit": comment_limit})
    posts, comments = _ordered(entries)
    if not posts and not comments:
        raise RuntimeError("No content found — is this a valid post URL?")

    marker_entries = ([posts[0]] if posts else []) + comments
    marker, next_cursor = _build_truncation_marker(marker_entries, _page_size())
    result = {
        "post": posts[0] if posts else None,
        "comment_count": len(comments),
        "comments": comments,
        "truncation": marker,
    }
    if next_cursor is not None:
        result["cursor"] = next_cursor
    return result


@mcp.tool()
async def search_reddit(
    query: str,
    subreddit: str | None = None,
    sort: str = "relevance",
    time_filter: str = "all",
    limit: int = 25,
    cursor: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Search Reddit discussions for real user opinions and experiences on a topic.

    Use this when the answer depends on what people who actually used or did the
    thing report: hands-on impressions, "is X worth it", failure stories,
    workarounds, recommendations, or where a community currently stands. Returns
    matching posts and their text; follow up with get_post to read the comments.

    Reddit's own search covers post titles and bodies, not comment text, so no
    results means Reddit did not surface it, not that nobody posted it.

    Args:
        query: Search terms.
        subreddit: Restrict to this subreddit (with or without "r/").
        sort: relevance, hot, top, new, comments.
        time_filter: hour, day, week, month, year, all.
        limit: Number of results (1-100).
        cursor: Opaque token from a previous call's `cursor`, to page through
            a result's full, untrimmed text.
    """
    limit = max(1, min(int(limit), 100))
    # The search feed mixes subreddit matches (t5) in with posts, so ask for a
    # few extra entries to still return `limit` posts after filtering them out.
    params = {"q": query, "sort": sort, "t": time_filter, "limit": min(100, limit + 5)}
    if subreddit:
        sub = subreddit.strip().removeprefix("r/").strip("/")
        params["restrict_sr"] = "1"
        url = f"{REDDIT_BASE}/r/{sub}/search.rss"
    else:
        url = f"{REDDIT_BASE}/search.rss"

    async def _fetch_full():
        _, full_entries = await _fetch_feed(url, params, limit=None)
        return [e for e in full_entries if e["type"] == "post"][:limit]

    expanded = await _paginate(ctx, cursor, _fetch_full)
    if expanded is not None:
        return expanded

    title, entries = await _fetch_feed(url, params)
    posts = [e for e in entries if e["type"] == "post"][:limit]
    marker, next_cursor = _build_truncation_marker(posts, _page_size())
    result = {"query": query, "count": len(posts), "results": posts, "truncation": marker}
    if next_cursor is not None:
        result["cursor"] = next_cursor
    return result


@mcp.tool()
async def fetch_json(
    url: str,
    cursor: str | None = None,
    ctx: Context | None = None,
) -> dict:
    """Read a Reddit feed the other tools do not cover: a user, multireddit, or domain.

    Use this only for Reddit URLs outside the usual shapes, such as
    /user/<name>, /r/<a>+<b>, or /domain/<site>. For browsing a subreddit,
    searching, or reading a thread, use browse_subreddit, search_reddit, or
    get_post instead.

    This is not a general web fetcher. URLs on other hosts are refused unless
    the operator sets REDDIT_MCP_ALLOW_ANY_URL=1.

    Args:
        url: A reddit.com URL.
        cursor: Opaque token from a previous call's `cursor`, to page through
            an entry's full, untrimmed text (Reddit-hosted URLs only).
    """
    parts = urlsplit(url.strip())

    if _is_reddit_host(parts.netloc):
        path = parts.path.rstrip("/")
        if not path.endswith(".rss"):
            path += "/.rss"
        target = urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))

        async def _fetch_full():
            _, full_entries = await _fetch_feed(target, limit=None)
            return full_entries

        expanded = await _paginate(ctx, cursor, _fetch_full)
        if expanded is not None:
            return expanded

        title, entries = await _fetch_feed(target)
        marker, next_cursor = _build_truncation_marker(entries, _page_size())
        result = {
            "url": target,
            "feed_title": title,
            "count": len(entries),
            "entries": entries,
            "truncation": marker,
        }
        if next_cursor is not None:
            result["cursor"] = next_cursor
        return result

    if not _allow_any_url():
        raise ValueError(
            f"{parts.netloc or url!s} is not a Reddit host, and this server only "
            "reads Reddit. To fetch other hosts, restart it with "
            "REDDIT_MCP_ALLOW_ANY_URL=1."
        )

    # Opted in: try the URL as given, then fall back to the .json convention.
    async with httpx.AsyncClient(
        timeout=TIMEOUT, follow_redirects=True, headers={"User-Agent": USER_AGENT}
    ) as client:
        resp = await _get_with_retry(client, url.strip())
        if resp.is_success:
            try:
                return {"url": str(resp.url), "data": resp.json()}
            except ValueError:
                pass  # not JSON — try the .json fallback below

        path = parts.path.rstrip("/")
        if not path.endswith(".json"):
            fallback = urlunsplit(
                (parts.scheme, parts.netloc, path + ".json", parts.query, "")
            )
            resp = await _get_with_retry(client, fallback)

    resp.raise_for_status()
    try:
        return {"url": str(resp.url), "data": resp.json()}
    except ValueError:
        # Structured trim fields (Task 2 pattern), not an inline "[truncated
        # ...]" suffix baked into the text.
        shown, total_chars, truncated = _trim_field(resp.text, 5000)
        return {
            "url": str(resp.url),
            "note": f"Response was not JSON (content-type: "
            f"{resp.headers.get('content-type', '')}).",
            "text": shown,
            "text_total_chars": total_chars,
            "text_truncated": truncated,
        }


def main() -> None:
    """Run the server over stdio for MCP clients."""
    mcp.run()


if __name__ == "__main__":
    main()
