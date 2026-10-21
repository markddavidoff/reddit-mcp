"""Structured trim fields on refined entries.

`_trim_field` is the single place that decides what gets shown vs. how much
text existed, so `_refine_entry` can report `text_total_chars` /
`text_truncated` instead of baking an inline "[truncated …]" string into the
text itself. No network: entries are built directly from Atom XML strings.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import server

ATOM = server.ATOM


def _atom_entry(content: str) -> ET.Element:
    xml_text = (
        '<entry xmlns="http://www.w3.org/2005/Atom">'
        "<id>t3_abc123</id>"
        "<title>A post</title>"
        f"<content>{content}</content>"
        "</entry>"
    )
    return ET.fromstring(xml_text)


def test_trim_field_short_returns_untrimmed():
    text = "short text"
    assert server._trim_field(text, 2000) == (text, len(text), False)


def test_trim_field_long_trims_and_reports_total():
    text = "x" * 3000
    shown, total, truncated = server._trim_field(text, 2000)
    assert len(shown) == 2000
    assert total == 3000
    assert truncated is True
    assert "[truncated" not in shown


def test_trim_field_none_limit_returns_full():
    text = "x" * 3000
    assert server._trim_field(text, None) == (text, len(text), False)


def test_trim_field_empty():
    assert server._trim_field(None, 2000) == (None, 0, False)
    assert server._trim_field("", 2000) == ("", 0, False)


def test_refine_entry_reports_truncation_fields():
    full_text = "y" * 2500
    entry = _atom_entry(full_text)
    refined = server._refine_entry(entry)
    assert refined["text_truncated"] is True
    assert refined["text_total_chars"] == len(full_text)
    assert len(refined["text"]) == server.TEXT_MAX
