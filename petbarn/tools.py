"""The two functions the assistant can call, and the schemas describing them.

Both return a plain dictionary and neither raises. A tool that throws leaves
the model with nothing to say and the viewer with an error box; a tool that
returns {"status": "not_found", ...} lets the model explain the problem in
words and offer what it does have.

Payload size is capped by construction rather than by hope. The free provider
tier allows eight thousand tokens a minute and a comparison resends a growing
conversation several times, so a tool that returned everything it knows would
exhaust the budget mid-answer, in front of whoever is watching.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from petbarn import analysis, aspects
from petbarn.catalogue import CATALOGUE_IDS
from petbarn.resolve import resolve
from petbarn.snapshot import Snapshot, get as get_snapshot

PRODUCT_URL = "https://www.petbarn.com.au/p/{slug}"

MAX_QUOTES = 8  # a comparison turn measures ~3.2k tokens, inside Groq's 8k/min
MAX_CAVEATS = 5
MAX_ASPECTS = 8


def _provenance(snap: Snapshot, slug: str | None = None) -> dict[str, Any]:
    provenance = {
        "source": "petbarn.com.au product pages and their Bazaarvoice review API",
        "captured_at": snap.captured_at.date().isoformat(),
        "snapshot_age_days": snap.age_days(),
    }
    if slug:
        provenance["product_url"] = PRODUCT_URL.format(slug=slug)
    return provenance


def _identify(query: Any, snap: Snapshot):
    """Resolve once and return both the outcome and, if it failed, the payload.

    Resolving twice used to throw away the resolver's note, so a fuzzy or
    partial match reached the model looking identical to an exact id.
    """
    found = resolve(query)
    if found.status == "resolved":
        return found, None

    return found, {
        "status": found.status,
        "requested": query,
        "note": found.note,
        "candidates": [
            {"id": cid, "name": snap.products[cid].display_name}
            for cid in found.candidates if cid in snap.products
        ],
        "instruction": (
            "Do not answer about a product that was not identified. Name the "
            "candidates above and ask which one was meant."
        ),
    }


def get_product_details(product: str) -> dict[str, Any]:
    """Specifications, pricing and description for one catalogue product."""
    snap = get_snapshot()

    found, problem = _identify(product, snap)
    if problem:
        return problem

    item = snap.products[found.catalogue_id]

    payload = {
        "status": "ok",
        "product": {
            "id": item.catalogue_id,
            "name": item.display_name,
            "brand": item.brand,
            "category": item.category,
            "pack_size": item.pack_size,
            "sku": item.sku,
            "price_aud": item.price_aud,
            "member_price_aud": item.member_price_aud,
            "in_stock": item.in_stock,
            "description": item.description,
            "rating": {
                "mean": _round(item.summary.mean_rating),
                "total_reviews": item.summary.total_reviews,
            },
        },
        "provenance": _provenance(snap, item.slug),
    }
    _note_substitution(payload, product, found)
    return payload


def _note_substitution(payload: dict[str, Any], requested: Any, found) -> None:
    """Tell the model when it did not get exactly what it asked for."""
    if found.note:
        payload["resolved_from"] = {"requested": str(requested)[:80], "note": found.note}


def _round(value: float | None, places: int = 2) -> float | None:
    """Ratings arrive from the API at full float precision. Nobody reads a star
    rating to fifteen decimal places, and every extra digit is a token."""
    return None if value is None else round(value, places)


def _sentiment_block(item, reviews, aspect: str | None, captured_on: date) -> dict[str, Any]:
    """Everything this project is willing to call sentiment.

    Three layers, weakest inference last. The star distribution is what people
    chose; the Quality and Value scores are what they chose on those axes
    specifically; the aspect rows compare the ratings of people who raised a
    subject against the ratings of those who did not. None of it is a guess
    about what anyone meant.
    """
    positive = [r for r in reviews if r.rating >= 4]
    mixed = [r for r in reviews if r.rating == 3]
    negative = [r for r in reviews if r.rating <= 2]

    # Anchored on the capture date. Against the wall clock the window shrinks
    # every day the snapshot ages and is empty ninety days after ingest, while
    # the field still calls itself the last 90 days.
    recent = analysis.recent(reviews, today=captured_on)
    averages = item.summary.secondary_averages

    quality_minus_value = None
    if averages.quality is not None and averages.value is not None:
        quality_minus_value = _round(averages.quality - averages.value)

    table = analysis.aspect_table(item, reviews)[: MAX_ASPECTS]

    # A topic the caller asked for keeps its own slot rather than competing with
    # the standard rows for the last one, and is reported even when nobody
    # raised it: "nobody mentioned this" is an answer, silence is not.
    requested: dict[str, Any] | None = None
    topic = aspects.clean_phrase(aspect)
    if topic:
        pattern = aspects.ad_hoc(topic)
        row = analysis.aspect_row(
            reviews, pattern, aspect=topic, label=f"{topic} (asked for)"
        ) if pattern else None
        if row:
            table = table[: MAX_ASPECTS - 1] + [row]
            requested = {"topic": topic, "mentions": row.mentions}
        else:
            requested = {"topic": topic, "mentions": 0,
                         "note": "no review mentions every word of this topic"}
    elif aspect:
        requested = {"topic": None,
                     "note": "the topic asked for contained nothing to match on"}

    return {
        # Two populations, and the difference matters: roughly half of all
        # reviews are a star rating with no text. Anything named a share or a
        # count below is over written reviews; mean_rating and the star
        # distribution are the retailer's own figures over every review.
        "counts_cover": "written reviews only",
        "written_reviews_analysed": len(reviews),
        "mean_rating_and_distribution_cover": "all reviews including ratings-only",
        "all_reviews_including_ratings_only": item.summary.total_reviews,
        "mean_rating": _round(item.summary.mean_rating),
        "star_distribution": {
            "5": item.summary.distribution.stars_5,
            "4": item.summary.distribution.stars_4,
            "3": item.summary.distribution.stars_3,
            "2": item.summary.distribution.stars_2,
            "1": item.summary.distribution.stars_1,
        },
        "share_positive_4_or_5": round(len(positive) / len(reviews), 3) if reviews else None,
        "share_mixed_3": round(len(mixed) / len(reviews), 3) if reviews else None,
        "share_negative_1_or_2": round(len(negative) / len(reviews), 3) if reviews else None,
        "would_recommend_rate": (
            round(item.summary.recommended
                  / (item.summary.recommended + item.summary.not_recommended), 3)
            if (item.summary.recommended + item.summary.not_recommended) else None
        ),
        # Scores the reviewers gave on these axes themselves, not inferred.
        "self_reported_scores": {
            "quality": _round(averages.quality),
            "value_for_money": _round(averages.value),
            "pet_satisfaction": _round(averages.pet_satisfaction),
            "quality_minus_value": quality_minus_value,
        },
        "requested_topic": requested,
        "by_aspect": [
            {
                "aspect": row.label,
                "mentions": row.mentions,
                "share_of_written_reviews": row.share_of_reviews,
                "mean_rating_of_mentioners": row.mean_rating,
                "difference_vs_other_reviewers": row.delta_vs_others,
                "low_confidence": row.low_confidence,
                **({"of_which_deny_it": row.denials} if row.denials else {}),
            }
            for row in table
        ],
        "last_90_days": {
            "window_ends": captured_on.isoformat(),
            "written_reviews": len(recent),
            "mean_rating": analysis.mean([r.rating for r in recent]) if recent else None,
        },
    }


def get_product_reviews_and_sentiment(
    product: str, aspect: str | None = None
) -> dict[str, Any]:
    """Ratings, sentiment and a selection of real reviews for one product."""
    snap = get_snapshot()

    found, problem = _identify(product, snap)
    if problem:
        return problem

    catalogue_id = found.catalogue_id
    item = snap.products[catalogue_id]
    reviews = snap.reviews[catalogue_id]

    if not reviews:
        return {
            "status": "no_reviews",
            "product": {"id": catalogue_id, "name": item.display_name},
            "provenance": _provenance(snap, item.slug),
        }

    quotes = analysis.sample(reviews, limit=MAX_QUOTES, critical_slots=3)
    caveats = analysis.harvest_caveats(reviews, limit=MAX_CAVEATS)
    critical_available = sum(1 for r in reviews if r.rating <= 3)

    payload = {
        "status": "ok",
        "product": {"id": catalogue_id, "name": item.display_name},
        "sentiment": _sentiment_block(item, reviews, aspect, snap.captured_at.date()),
        "quotes": [
            {
                "rating": q.rating,
                "date": q.submitted_on.isoformat(),
                "verified_purchaser": q.verified_purchaser,
                "found_helpful_by": q.helpful_up,
                # Emitted whole. Ingest already capped bodies at 600 characters
                # with a word-boundary cut and an ellipsis; slicing again here
                # once ended a five star review mid-sentence at "very bad
                # allergy reaction", where the rest explained the reaction was
                # to a competitor's product and this one fixed it.
                "text": q.body,
            }
            for q in quotes
        ],
        "caveats_from_positive_reviews": [
            {"rating": c.rating, "date": c.submitted_on.isoformat(), "text": c.clause}
            for c in caveats
        ],
        # Without this the model will describe a deliberately skewed sample as
        # though it were representative, which is the most likely way this
        # answers something untrue.
        "coverage": {
            "quotes_shown": len(quotes),
            "of_written_reviews": len(reviews),
            "selection": (
                "Not a random sample. Slots are reserved for reviews rated 3 or "
                "below because they are scarce, so the quotes are deliberately "
                "more critical than the corpus. Use the shares above for "
                "proportions and the quotes only as illustration."
            ),
            "written_reviews_rated_3_or_below": critical_available,
            "quotes_date_range": [
                min(q.submitted_on for q in quotes).isoformat(),
                max(q.submitted_on for q in quotes).isoformat(),
            ],
            "caveats_note": (
                "Reservations expressed inside 4 and 5 star reviews. In a corpus "
                "this positive most criticism appears this way rather than as a "
                "low rating."
            ),
        },
        "provenance": _provenance(snap, item.slug),
    }
    _note_substitution(payload, product, found)
    return payload


# Declarations sent to the model. The product argument is an enum so a valid id
# is the only thing the model can legitimately produce; the resolver still runs,
# because not every provider enforces an enum.
SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_product_details",
            "description": (
                "Specifications, price, member price, stock and description for one "
                "product in the nine-product Petbarn catalogue."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "enum": list(CATALOGUE_IDS),
                        "description": "Catalogue id of the product.",
                    }
                },
                "required": ["product"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_reviews_and_sentiment",
            "description": (
                "Ratings, sentiment breakdown and real customer review extracts for "
                "one product. Returns star distribution, the shares of positive, "
                "mixed and negative reviews, the Quality and Value scores customers "
                "gave themselves, a per-topic breakdown comparing the ratings of "
                "people who raised each topic against everyone else, quotes, and "
                "reservations mined from otherwise positive reviews. Call once per "
                "product; to compare two products, call it twice."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "enum": list(CATALOGUE_IDS),
                        "description": "Catalogue id of the product.",
                    },
                    "aspect": {
                        "type": "string",
                        "description": (
                            "Optional topic to analyse in addition to the standard "
                            "ones, in the customer's own words, for example "
                            "'sensitive stomach' or 'puppies'. Omit unless the "
                            "question asks about something specific."
                        ),
                    },
                },
                "required": ["product"],
                "additionalProperties": False,
            },
        },
    },
]

REGISTRY = {
    "get_product_details": get_product_details,
    "get_product_reviews_and_sentiment": get_product_reviews_and_sentiment,
}
