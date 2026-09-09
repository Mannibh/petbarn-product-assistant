"""Cleaning applied to review text at ingest, once, before it is stored.

Reviews are written by strangers and end up inside a language model's prompt.
That makes them untrusted input, and the cheapest place to deal with untrusted
input is at the boundary where it enters the system rather than at every point
that later reads it.

Nothing here deletes a review. Suspicious text is flagged and kept: written
criticism is scarce in this corpus, and quietly dropping reviews would bias
every answer the assistant gives.
"""

from __future__ import annotations

import html
import re
import unicodedata

# Characters that render as nothing but survive a copy-paste: zero-width spaces
# and joiners, the word joiner, and the bidirectional overrides that can make
# text display in a different order than it is stored. Written as escapes: with
# the codepoints themselves this line looks like an empty pair of brackets, and
# a reader cannot tell what it matches or check it in a diff.
INVISIBLE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff]")

# C0 and C1 control codes, keeping tab and newline.
CONTROLS = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]")

HTML_TAG = re.compile(r"<[^>]{1,200}>")

WHITESPACE = re.compile(r"[ \t]{2,}")

# Phrases that appear when someone is talking to the model rather than to other
# shoppers. Deliberately broad: a false positive costs a line in a report, and a
# false negative costs little either, because the tools are read-only. This
# exists to be counted and reported on, not to be relied upon as a defence.
INJECTION_MARKERS = re.compile(
    r"""
    ignore\s+(all\s+|any\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?)
  | disregard\s+(the\s+)?(previous|prior|above|system)
  | system\s+prompt
  | you\s+are\s+now\s+(a|an)\b
  | new\s+instructions?\s*:
  | <\|(im_start|im_end|endoftext)\|>
  | \[/?INST\]
  | ^\s*(assistant|system)\s*:
    """,
    re.IGNORECASE | re.VERBOSE | re.MULTILINE,
)

# Long enough for the longest review anyone actually reads to the end of, short
# enough that a dozen quotes fit inside a free tier's per-minute token budget
# without crowding out the statistics they are there to illustrate.
MAX_BODY_CHARS = 600


def clean_review_text(raw: str | None, *, max_chars: int = MAX_BODY_CHARS) -> tuple[str, bool]:
    """Return cleaned text and whether it tripped an injection marker.

    Order matters twice over. Unicode is normalised first so visually identical
    characters written different ways collapse to one form. Then the invisible
    and control sweeps run again after entity decoding, because "&#8203;"
    decodes into exactly the zero-width space the first sweep removed, and a
    review written that way would otherwise reach the marker check with its
    wording broken up and its meaning intact.
    """
    if not raw:
        return "", False

    text = unicodedata.normalize("NFKC", raw)
    text = INVISIBLE.sub("", text)
    text = CONTROLS.sub("", text)

    text = html.unescape(text)
    text = INVISIBLE.sub("", text)
    text = CONTROLS.sub("", text)

    text = HTML_TAG.sub(" ", text)
    text = WHITESPACE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    flagged = bool(INJECTION_MARKERS.search(text))

    if len(text) > max_chars:
        # Trim at a word boundary so a quote never ends mid-word.
        cut = text[:max_chars].rsplit(" ", 1)[0]
        text = f"{cut}..."

    return text, flagged
