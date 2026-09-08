"""The analysis layer produces every number the assistant is allowed to state,
so a quiet bug here becomes a confidently wrong answer."""

from __future__ import annotations

from petbarn import aspects
from petbarn.analysis import aspect_table, harvest_caveats, recent, sample
from petbarn.models import Product, ReviewSummary, RatingDistribution, AspectRatings
from tests.conftest import make_review

from datetime import date


def dog_food() -> Product:
    return Product(
        catalogue_id="test-product", display_name="Test Food", brand="Test",
        category="dog_food", sku="1", slug="test", price_aud=10.0,
        in_stock=True, description="x",
        summary=ReviewSummary(
            total_reviews=0, text_reviews=0, mean_rating=5,
            distribution=RatingDistribution(stars_1=0, stars_2=0, stars_3=0, stars_4=0, stars_5=0),
            secondary_averages=AspectRatings(), recommended=0, not_recommended=0,
        ),
    )


def test_word_boundaries_prevent_false_matches():
    """"priceless" is not about price, and this is the whole reason the patterns
    are built with boundaries rather than plain substring search."""
    pattern = aspects.for_category("dog_food")["price_value"]
    assert pattern.search("the price went up")
    assert not pattern.search("this food is priceless to us")


def test_aspect_delta_compares_mentioners_against_everyone_else():
    """Not against the whole corpus. Including the mentioners in their own
    baseline flattens the difference in proportion to how many of them there
    are, which reached threefold on the real data."""
    reviews = (
        [make_review(rating=2, body="the price is far too high", review_id=f"a{i}") for i in range(5)]
        + [make_review(rating=5, body="dog loves the taste", review_id=f"b{i}") for i in range(15)]
    )
    row = {r.aspect: r for r in aspect_table(dog_food(), reviews)}["price_value"]
    assert row.mentions == 5
    assert row.delta_vs_others == -3.0


def test_thin_aspect_rows_are_marked_low_confidence():
    """At three mentions one dissenter moves the mean by more than a star. The
    largest delta in the real corpus was built from exactly three reviews."""
    reviews = (
        [make_review(rating=1, body="the price is outrageous", review_id="a1")]
        + [make_review(rating=5, body="dog loves the taste", review_id=f"b{i}") for i in range(30)]
    )
    row = {r.aspect: r for r in aspect_table(dog_food(), reviews)}["price_value"]
    assert row.mentions == 1
    assert row.low_confidence


def test_words_from_the_product_name_are_not_counted_as_an_aspect():
    """"Chicken" in an ingredients lexicon counts every review that simply names
    Pro Plan Chicken, including ones about whether the cat liked it."""
    from petbarn import aspects
    name = "Love Em Air Dried Beef Liver Dog Treats"
    generic = aspects.for_category("dog_treat")
    specific = aspects.for_product("dog_treat", name)

    # "liver" is a legitimate ingredient word and it is also this product's name
    assert generic["ingredients"].search("plenty of liver in these")
    assert not specific["ingredients"].search("plenty of liver in these")
    # a term unrelated to the name is untouched
    assert specific["ingredients"].search("no nasty preservatives")


def test_strong_marker_returns_the_whole_sentence_not_a_fragment():
    """Asserting equality, not membership. The original test checked that
    "tears open" appeared somewhere in the clause, which stayed green while
    every clause the function produced was a dangling fragment."""
    reviews = [make_review(
        rating=5,
        body="Excellent food. My only complaint is that the bag tears open too easily.",
    )]
    caveats = harvest_caveats(reviews)
    assert len(caveats) == 1
    assert caveats[0].clause == "My only complaint is that the bag tears open too easily"


def test_weak_marker_with_a_complaint_is_kept():
    reviews = [make_review(
        rating=5, body="Great food but the price keeps going up every year.")]
    assert harvest_caveats(reviews)[0].clause == "Great food but the price keeps going up every year"


def test_negated_complaint_is_not_a_reservation():
    """"No issues" contains the complaint word and means the opposite. Seven of
    Pro Plan's eight original caveats were compliments of this shape."""
    reviews = [
        make_review(rating=5, body="Great pouches but she had no issues finishing it.", review_id="1"),
        make_review(rating=5, body="Tried it but never had a problem with this food.", review_id="2"),
    ]
    assert harvest_caveats(reviews) == []


def test_an_unnegated_issue_is_still_a_reservation():
    """The veto must not swallow the genuine case it sits next to."""
    reviews = [make_review(rating=5, body="Lovely food but the only issue is the smell.")]
    assert len(harvest_caveats(reviews)) == 1


def test_a_complaint_in_a_later_sentence_does_not_admit_an_earlier_clause():
    """The hint has to be scoped to the marker's own sentence. Searching the
    rest of the review admitted praise because something unrelated further down
    happened to complain."""
    reviews = [make_review(
        rating=5,
        body="Fussy cat but he gobbled it up. Delivery was slow and the box was damaged.",
    )]
    assert harvest_caveats(reviews) == []


def test_weak_marker_introducing_praise_is_not_a_caveat():
    """The failure the first version made: "but" and "only" introduce praise at
    least as often as they introduce a reservation."""
    reviews = [
        make_review(rating=5, body="Expensive but it is absolutely worth every cent.", review_id="1"),
        make_review(rating=5, body="This is the only brand our dog will happily eat.", review_id="2"),
    ]
    assert harvest_caveats(reviews) == []


def test_critical_reviews_are_never_mined_for_caveats():
    """A low rating is already criticism and gets quoted directly."""
    reviews = [make_review(rating=2, body="Rubbish, but the delivery was slow too.")]
    assert harvest_caveats(reviews) == []


def test_sampler_reserves_slots_for_scarce_critical_reviews():
    """96% of the real corpus is four or five stars, so a proportionate sample
    is all praise and a question about drawbacks gets a useless answer."""
    reviews = (
        [make_review(rating=5, body="great", review_id=f"p{i}", helpful=i) for i in range(50)]
        + [make_review(rating=2, body="poor", review_id="c1", days_ago=1)]
    )
    chosen = sample(reviews, limit=8, critical_slots=3)
    assert len(chosen) == 8
    assert any(q.rating <= 3 for q in chosen)


def test_sample_never_returns_more_than_the_limit():
    """Reserved critical slots are clipped against the limit too: asking for two
    quotes must not return three because three were reserved."""
    reviews = (
        [make_review(rating=2, body="poor", review_id=f"c{i}", days_ago=i) for i in range(5)]
        + [make_review(rating=5, body="great", review_id=f"p{i}") for i in range(10)]
    )
    assert len(sample(reviews, limit=2, critical_slots=3)) == 2
    assert sample(reviews, limit=0, critical_slots=3) == []


def test_positive_quotes_are_ordered_by_helpfulness():
    reviews = [make_review(rating=5, review_id=f"p{i}", helpful=i) for i in range(10)]
    chosen = sample(reviews, limit=3)
    assert [q.helpful_up for q in chosen] == [9, 8, 7]


def test_recent_window_filters_by_date():
    reviews = [make_review(review_id="new", days_ago=10), make_review(review_id="old", days_ago=400)]
    got = recent(reviews, today=date(2026, 9, 8), days=90)
    assert [r.review_id for r in got] == ["new"]
