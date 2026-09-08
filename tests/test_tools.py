"""The two functions the model calls.

Two properties matter more than the contents: they never raise, and their
output stays inside the token budget a free provider tier allows.
"""

from __future__ import annotations

import json

import pytest

from petbarn.catalogue import CATALOGUE_IDS
from petbarn.tools import (
    MAX_QUOTES,
    REGISTRY,
    SCHEMAS,
    get_product_details,
    get_product_reviews_and_sentiment,
)


def approx_tokens(payload) -> int:
    return len(json.dumps(payload, separators=(",", ":"))) // 4


@pytest.mark.parametrize("bad", ["", None, "hills science diet", "dog food", "!!!", "12345"])
@pytest.mark.parametrize("tool", list(REGISTRY.values()), ids=list(REGISTRY))
def test_tools_never_raise(tool, bad):
    """A tool that throws leaves the model with nothing to say and the viewer
    with an error box. Every failure comes back as a status instead."""
    result = tool(bad)
    assert result["status"] in {"ok", "ambiguous", "not_found", "no_reviews"}


def test_an_unidentified_product_offers_candidates_and_forbids_guessing():
    result = get_product_reviews_and_sentiment("hills science diet")
    assert result["status"] == "not_found"
    assert result["candidates"]
    assert "not identified" in result["instruction"]


def test_details_carry_both_prices_and_provenance():
    result = get_product_details("royal-canin-maxi")
    product = result["product"]
    assert product["price_aud"] > product["member_price_aud"]
    assert result["provenance"]["product_url"].endswith("royal-canin-maxi-adult-dog-food-15kg")
    assert result["provenance"]["captured_at"]


def test_the_reviews_tool_returns_a_sentiment_block():
    """The brief asks this tool for sentiment, so the payload has to contain it
    rather than leave the model to infer it from raw text."""
    sentiment = get_product_reviews_and_sentiment("black-hawk-lamb-rice")["sentiment"]
    for field in ("star_distribution", "share_positive_4_or_5", "share_negative_1_or_2",
                  "self_reported_scores", "by_aspect", "would_recommend_rate"):
        assert field in sentiment
    assert sentiment["self_reported_scores"]["quality_minus_value"] is not None


def test_sentiment_numbers_are_arithmetically_true():
    """Nothing else in the suite checks a sentiment figure, so a corrupted
    calculation would ship silently. These are recomputed from the snapshot."""
    from petbarn.snapshot import get as snapshot

    snap = snapshot()
    item = snap.products["breeders-choice-litter"]
    written = snap.reviews["breeders-choice-litter"]
    s = get_product_reviews_and_sentiment("breeders-choice-litter")["sentiment"]

    assert s["written_reviews_analysed"] == len(written)
    assert sum(s["star_distribution"].values()) == s["all_reviews_including_ratings_only"]

    positive = sum(1 for r in written if r.rating >= 4)
    assert s["share_positive_4_or_5"] == round(positive / len(written), 3)
    assert round(
        s["share_positive_4_or_5"] + s["share_mixed_3"] + s["share_negative_1_or_2"], 2
    ) == 1.0

    scores = s["self_reported_scores"]
    assert scores["quality_minus_value"] == round(
        scores["quality"] - scores["value_for_money"], 2)

    for row in s["by_aspect"]:
        assert 0 < row["share_of_written_reviews"] <= 1
        assert 1 <= row["mean_rating_of_mentioners"] <= 5
        assert row["low_confidence"] == (row["mentions"] < 5)


def test_quotes_are_never_cut_mid_word():
    """A five star review once arrived sliced at "very bad allergy reaction",
    where the rest explained the reaction was to a competitor's product."""
    for pid in ("breeders-choice-litter", "black-hawk-lamb-rice", "prime100-roo-roll"):
        for quote in get_product_reviews_and_sentiment(pid)["quotes"]:
            assert quote["text"] == quote["text"].rstrip()
            assert quote["text"].endswith(("...", ".", "!", "?")) or " " not in quote["text"][-1]


def test_the_sample_declares_that_it_is_skewed():
    """Without this the model describes a deliberately critical-weighted sample
    as representative, which is the most likely way it says something untrue."""
    coverage = get_product_reviews_and_sentiment("breeders-choice-litter")["coverage"]
    assert coverage["quotes_shown"] <= MAX_QUOTES
    assert coverage["of_written_reviews"] > coverage["quotes_shown"]
    assert "Not a random sample" in coverage["selection"]
    assert coverage["written_reviews_rated_3_or_below"] > 0


def test_ratings_are_rounded_for_reading():
    sentiment = get_product_reviews_and_sentiment("black-hawk-lamb-rice")["sentiment"]
    assert sentiment["mean_rating"] == round(sentiment["mean_rating"], 2)
    assert sentiment["self_reported_scores"]["quality"] == round(
        sentiment["self_reported_scores"]["quality"], 2)


def test_a_comparison_turn_fits_inside_the_free_tier_budget():
    """Groq's free tier allows 8,000 tokens a minute and a comparison resends a
    growing conversation, so one turn must leave room for several rounds."""
    turn = [SCHEMAS,
            get_product_reviews_and_sentiment("black-hawk-lamb-rice"),
            get_product_reviews_and_sentiment("prime100-roo-roll")]
    assert approx_tokens(turn) < 4500


def test_schema_enum_matches_the_catalogue():
    """The enum is the model's only legitimate source of product ids; a drift
    between it and the catalogue would invite hallucinated arguments."""
    for schema in SCHEMAS:
        product = schema["function"]["parameters"]["properties"]["product"]
        assert tuple(product["enum"]) == CATALOGUE_IDS


def test_an_ad_hoc_aspect_is_analysed_alongside_the_standard_ones():
    """What stops the fixed vocabulary being a ceiling: a question about
    something nobody anticipated gets the same statistics."""
    result = get_product_reviews_and_sentiment("black-hawk-lamb-rice", aspect="puppy")
    sentiment = result["sentiment"]
    assert sentiment["requested_topic"]["topic"] == "puppy"
    assert any("puppy" in row["aspect"] for row in sentiment["by_aspect"])


def test_a_requested_topic_keeps_its_slot_on_a_product_with_many_aspects():
    """Cat litter has eight standard aspects, so an ad-hoc row used to be the
    one truncated away after being computed."""
    sentiment = get_product_reviews_and_sentiment(
        "breeders-choice-litter", aspect="sensitive skin")["sentiment"]
    assert sentiment["requested_topic"]["topic"] == "sensitive skin"


def test_a_multi_word_topic_requires_every_word():
    """Matching "sensitive stomach" as either word counts every mention of a
    stomach and reports it as people discussing sensitive stomachs."""
    both = get_product_reviews_and_sentiment(
        "black-hawk-lamb-rice", aspect="sensitive stomach")["sentiment"]["requested_topic"]
    just_one = get_product_reviews_and_sentiment(
        "black-hawk-lamb-rice", aspect="stomach")["sentiment"]["requested_topic"]
    assert both["mentions"] < just_one["mentions"]


def test_an_oversized_topic_is_bounded_before_it_is_echoed():
    sentiment = get_product_reviews_and_sentiment(
        "black-hawk-lamb-rice", aspect="price " + "A" * 400)["sentiment"]
    assert len(sentiment["requested_topic"]["topic"]) <= 40


def test_a_partial_product_match_is_disclosed_to_the_model():
    """We stock no Black Hawk puppy food. Answering about the adult food without
    saying so is the confident wrong answer this exists to prevent."""
    result = get_product_reviews_and_sentiment("black hawk puppy food")
    assert result["status"] == "ok"
    assert "puppy" in result["resolved_from"]["note"]


@pytest.mark.parametrize("junk", [5, ["x"], {"a": 1}, 3.5])
def test_a_non_string_product_does_not_raise(junk):
    assert get_product_details(junk)["status"] == "not_found"
    assert get_product_reviews_and_sentiment(junk)["status"] == "not_found"
