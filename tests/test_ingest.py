"""The validation gate, which decides whether a scrape is allowed to ship.

No network: check() takes objects, not URLs. This is the last thing standing
between a partial scrape and a committed snapshot, so it is worth more coverage
than the fetching around it.
"""

from __future__ import annotations

from datetime import date

from petbarn.catalogue import CATALOGUE
from petbarn.ingest import MIN_TEXT_REVIEWS, REVIEW_SHORTFALL_TOLERANCE, check
from petbarn.models import AspectRatings, Product, RatingDistribution, ReviewSummary
from tests.conftest import make_review


def product(catalogue_id="test-product", *, price=10.0, description="real copy",
            total=200, text=120, stars_3=5) -> Product:
    return Product(
        catalogue_id=catalogue_id, display_name="Test", brand="Test",
        category="dog_food", sku="1", slug="test", price_aud=price,
        in_stock=True, description=description,
        summary=ReviewSummary(
            total_reviews=total, text_reviews=text, mean_rating=4.5,
            distribution=RatingDistribution(
                stars_1=0, stars_2=0, stars_3=stars_3, stars_4=0,
                stars_5=total - stars_3),
            secondary_averages=AspectRatings(), recommended=100, not_recommended=1,
        ),
    )


def reviews(n, catalogue_id="test-product", rating=5):
    return [make_review(rating=rating, review_id=f"{catalogue_id}-{i}",
                        catalogue_id=catalogue_id) for i in range(n)]


def test_a_complete_scrape_passes():
    p = product()
    assert check([p], {"test-product": reviews(120)}, expected=1) == []


def test_a_missing_product_is_rejected():
    """A snapshot where some products have data and others do not is worse than
    none, because nothing downstream can tell the difference."""
    problems = check([], {}, expected=len(CATALOGUE))
    assert any("expected 9 products" in p for p in problems)


def test_a_subset_run_is_judged_on_its_own_terms():
    """The --only flag exists for quick testing and used to fetch everything
    and then always abort, because the gate compared against all nine."""
    p = product()
    assert check([p], {"test-product": reviews(120)}, expected=1) == []


def test_a_product_with_no_price_is_rejected():
    p = product(price=0)
    assert any("no price" in x for x in check([p], {"test-product": reviews(120)}, expected=1))


def test_a_product_with_no_description_is_rejected():
    p = product(description="")
    assert any("no description" in x for x in check([p], {"test-product": reviews(120)}, expected=1))


def test_too_few_written_reviews_is_rejected():
    p = product(text=MIN_TEXT_REVIEWS - 10)
    problems = check([p], {"test-product": reviews(MIN_TEXT_REVIEWS - 10)}, expected=1)
    assert any("written reviews, need" in x for x in problems)


def test_a_truncated_review_fetch_is_rejected():
    """The failure this exists for: paging stops early, and without this the
    snapshot looks complete and simply has less to say."""
    p = product(text=500)
    problems = check([p], {"test-product": reviews(120)}, expected=1)
    assert any("collected 120 written reviews but the source reports 500" in x
               for x in problems)


def test_a_small_shortfall_is_tolerated():
    """The source occasionally reports a written review whose body is null, so
    the count can sit one or two under what it advertises."""
    p = product(text=120 + REVIEW_SHORTFALL_TOLERANCE - 1)
    assert check([p], {"test-product": reviews(120)}, expected=1) == []


def test_duplicate_review_ids_are_rejected():
    """Offset paging over a newest-first sort repeats a row at the boundary if
    a review is posted mid-scrape."""
    duplicated = reviews(120)
    duplicated[50] = duplicated[49]
    problems = check([product()], {"test-product": duplicated}, expected=1)
    assert any("duplicate review ids" in x for x in problems)


def test_star_counts_that_disagree_with_the_total_are_rejected():
    p = product(total=200)
    p = p.model_copy(update={"summary": p.summary.model_copy(
        update={"distribution": RatingDistribution(
            stars_1=1, stars_2=1, stars_3=1, stars_4=1, stars_5=1)})})
    problems = check([p], {"test-product": reviews(120)}, expected=1)
    assert any("star counts total" in x for x in problems)
