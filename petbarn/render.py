"""Making model output safe to put on a page.

The chat window renders markdown, and everything shown in it has passed through
customer reviews written by strangers. A review containing an image reference
would make the reader's browser fetch that image the moment the answer appears,
from whichever server the review author chose, without the reader doing
anything. That is the one sharp edge in an app whose tools are read-only, and
it is closed here rather than trusted to the model.

Ingest already cleans the review text itself. This is the second half: the
model composes an answer from that text, and what the model writes is not what
the tool returned.
"""

from __future__ import annotations

import re

# ![anything](anywhere) - the syntax that makes a browser fetch on render.
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# [text](url) - kept when it points at the retailer, otherwise reduced to text.
LINK = re.compile(r"\[([^\]]*)\]\(\s*([^)\s]+)[^)]*\)")

BARE_URL = re.compile(r"(?:https?://|www\.)[^\s<>\)]+", re.IGNORECASE)

# [1]: http://host/x - a reference definition, which a surviving [ref][1] would
# resolve against. Removed outright rather than rewritten.
LINK_DEFINITION = re.compile(r"^[ \t]*\[[^\]]+\]:[ \t]*\S+.*$", re.MULTILINE)

CONTROLS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")

ALLOWED_HOSTS = ("petbarn.com.au",)


def _allowed(url: str) -> bool:
    host = re.sub(r"^(https?:)?//", "", url.lower()).split("/")[0].split("@")[-1]
    return any(host == h or host.endswith("." + h) for h in ALLOWED_HOSTS)


def safe_markdown(text: str) -> str:
    """Strip anything that would make the page fetch from somewhere we do not control."""
    if not text:
        return ""

    text = CONTROLS.sub("", text)
    text = IMAGE.sub("", text)
    # Enumerating hostile syntax is a losing game: CommonMark writes an image
    # four ways and the pattern above sees two of them. Anything still holding
    # an exclamation mark in front of a bracket is defused instead, so no image
    # construct can survive whatever shape it arrived in.
    text = text.replace("![", "!\\[")
    text = LINK_DEFINITION.sub("", text)
    text = LINK.sub(
        lambda m: f"[{m.group(1)}]({m.group(2)})" if _allowed(m.group(2)) else m.group(1),
        text,
    )
    text = BARE_URL.sub(lambda m: m.group(0) if _allowed(m.group(0)) else "[link removed]", text)

    return text.strip()
