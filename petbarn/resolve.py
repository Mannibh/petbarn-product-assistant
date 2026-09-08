"""Turns whatever arrives in a tool call into one of the nine catalogue ids.

The model is told the catalogue and the argument is declared as an enum, so
most of the time it passes an exact id and this does nothing. This exists for
the rest of the time, and it is not belt and braces: enum enforcement is a
promise only some providers keep, and Anthropic's OpenAI-compatible endpoint
documents that it ignores the strict flag entirely.

The rule is that it never guesses silently. A near miss comes back as an
ambiguity with candidates, so the assistant can ask, rather than confidently
answering about the wrong product.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Literal

from petbarn.catalogue import BY_ID, CATALOGUE

Status = Literal["resolved", "ambiguous", "not_found"]

# Phrases that name a shelf rather than a product, mapped to the categories they
# cover. Someone asking about "the dog food" has not said which of the three, so
# the answer is a question back. But "cat litter" names a category we stock
# exactly one of, and refusing to answer that would be pedantic, so a category
# term resolves when only one product sits in it.
GENERIC: dict[str, tuple[str, ...]] = {
    "food": ("dog_food", "cat_food"),
    "pet food": ("dog_food", "cat_food"),
    "dog food": ("dog_food",),
    "dry food": ("dog_food",),
    "kibble": ("dog_food",),
    "cat food": ("cat_food",),
    "wet food": ("cat_food",),
    "treat": ("dog_treat",),
    "treats": ("dog_treat",),
    "dog treat": ("dog_treat",),
    "dog treats": ("dog_treat",),
    "litter": ("cat_litter",),
    "cat litter": ("cat_litter",),
    "flea": ("parasite_control",),
    "tick": ("parasite_control",),
    "worm": ("parasite_control",),
    "flea chew": ("parasite_control",),
    "flea treatment": ("parasite_control",),
    "chew": ("parasite_control",),
    "chews": ("parasite_control",),
}

MIN_SIMILARITY = 0.62

# Ordinary sentence words. Left over after a product name is matched they carry
# no information, so they are not treated as something the query asked for.
FILLER = {
    "the", "a", "an", "and", "or", "of", "for", "from", "with", "about", "on",
    "in", "to", "is", "are", "it", "this", "that", "what", "how", "do", "does",
    "people", "say", "saying", "tell", "me", "you", "please", "can", "could",
    "review", "reviews", "rating", "ratings", "price", "cost", "buy", "good",
    "bad", "best", "worst", "like", "think", "any", "some", "more", "much",
}


@dataclass(frozen=True)
class Resolution:
    status: Status
    catalogue_id: str | None
    candidates: tuple[str, ...]
    note: str


def _normalise(text: str) -> str:
    text = text.lower().replace("&", " and ")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _searchable() -> dict[str, str]:
    """Every phrase that identifies a product, mapped to its catalogue id."""
    index: dict[str, str] = {}
    for entry in CATALOGUE:
        for key in (entry.catalogue_id, entry.sku, entry.slug, entry.display_name,
                    *entry.aliases):
            index[_normalise(key)] = entry.catalogue_id
    return index


INDEX = _searchable()


def _ignored_words(text: str, catalogue_id: str) -> str:
    """Words in the query that the matched product does not account for.

    "Black Hawk puppy food" contains a product we stock and a variant we do not.
    Matching the brand and saying nothing would answer confidently about the
    adult food, so the leftovers are reported and the assistant can qualify it.
    """
    entry = BY_ID[catalogue_id]
    known = set(_normalise(
        " ".join((entry.display_name, entry.brand, entry.category.replace("_", " "),
                  entry.pack_size or "", *entry.aliases))
    ).split())

    leftover = [w for w in text.split()
                if w not in known and w not in FILLER and len(w) > 2]
    if not leftover:
        return ""
    return f"matched on the product name; these words were not used: {', '.join(leftover)}"


def _category_match(text: str) -> tuple[str, ...] | None:
    """Catalogue ids covered by a shelf term in the query, or None.

    Longest phrase first: "cat food" has to be tested before "food", or the
    shorter one matches inside it and answers about dog food.
    """
    for phrase in sorted(GENERIC, key=len, reverse=True):
        if re.search(rf"\b{re.escape(phrase)}\b", text):
            return tuple(
                e.catalogue_id for e in CATALOGUE if e.category in GENERIC[phrase]
            )
    return None


def resolve(query: str | None) -> Resolution:
    """Identify a product, or say honestly that it could not be identified.

    Specific beats general throughout. "The dog food from Black Hawk" names a
    product and a shelf, and the product wins; only a query with nothing
    specific in it falls through to the category rules.
    """
    # The argument comes from a model, so it is not guaranteed to be a string.
    if not isinstance(query, str) or not query.strip():
        return Resolution("not_found", None, tuple(BY_ID), "no product was named")

    # Exact catalogue id: the common case, and the one the enum should produce.
    if query in BY_ID:
        return Resolution("resolved", query, (), "")

    text = _normalise(query)

    if text in INDEX:
        return Resolution("resolved", INDEX[text], (), "")

    # A known name appearing inside the query is a strong signal: "the dog food
    # from black hawk" names a product and a shelf, and the product wins.
    contains = {cid for phrase, cid in INDEX.items()
                if len(phrase) > 3 and phrase in text}
    if len(contains) == 1:
        matched = contains.pop()
        return Resolution("resolved", matched, (), _ignored_words(text, matched))
    if len(contains) > 1:
        return Resolution(
            "ambiguous", None, tuple(sorted(contains)),
            f"'{query}' matches more than one product",
        )

    in_category = _category_match(text)
    if in_category:
        if len(in_category) == 1:
            return Resolution("resolved", in_category[0], (), "")
        return Resolution(
            "ambiguous", None, in_category,
            f"'{query}' covers {len(in_category)} products in the catalogue",
        )

    # The other direction, checked only once nothing more specific has matched:
    # a short query sitting inside a longer alias is weak evidence, because
    # "dog food" is inside "maxi breed dog food" without meaning that product.
    within = {cid for phrase, cid in INDEX.items()
              if len(text) > 3 and text in phrase}
    if len(within) == 1:
        return Resolution("resolved", within.pop(), (), "")
    if len(within) > 1:
        return Resolution(
            "ambiguous", None, tuple(sorted(within)),
            f"'{query}' matches more than one product",
        )

    close = difflib.get_close_matches(text, INDEX, n=3, cutoff=MIN_SIMILARITY)
    suggested = tuple(dict.fromkeys(INDEX[m] for m in close))
    if len(suggested) == 1:
        return Resolution("resolved", suggested[0], (), f"read '{query}' as the closest match")
    if suggested:
        return Resolution("ambiguous", None, suggested, f"'{query}' could be several products")

    return Resolution(
        "not_found", None, tuple(BY_ID),
        f"'{query}' is not in this nine-product catalogue",
    )
