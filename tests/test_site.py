"""Parser tests against structured data captured from the real product pages.

The fixtures are the three shapes Petbarn actually serves. A parser written
against only the first returns nothing for the other two, without erroring.
"""

from __future__ import annotations

import pytest

from petbarn.site import collect_products, parse_product_page


def test_simple_product(fixture_html):
    facts = parse_product_page(fixture_html("product_simple.html"), sku="30299")
    assert facts["sku"] == "30299"
    assert facts["brand"] == "ROYAL CANIN"
    assert facts["price_aud"] == 233.99
    assert facts["in_stock"] is True
    assert facts["review_count"] == 254


def test_variant_group_carrying_its_own_rating(fixture_html):
    facts = parse_product_page(fixture_html("product_group_rated.html"), sku="117874")
    assert facts["sku"] == "117874"
    assert facts["price_aud"] == 38.49
    assert facts["rating_value"] is not None


def test_variant_group_with_no_sku_or_rating_of_its_own(fixture_html):
    """The shape that returns nothing to a parser expecting type Product: the
    group carries no sku and no rating, only a list of variants."""
    html = fixture_html("product_group_bare.html")
    top = collect_products(html)[0]
    assert top["@type"] == "ProductGroup"
    assert top.get("sku") is None

    facts = parse_product_page(html, sku="127959")
    assert facts["sku"] == "127959"
    assert facts["price_aud"] == 200.99


def test_member_price_is_the_loyalty_tier_not_the_shelf_price(fixture_html):
    """priceSpecification holds both prices under the same type, told apart only
    by validForMemberTier. Matching on type alone returns the shelf price."""
    facts = parse_product_page(fixture_html("product_simple.html"), sku="30299")
    assert facts["member_price_aud"] == 159
    assert facts["member_price_aud"] < facts["price_aud"]


def test_unknown_sku_raises_rather_than_returning_empty(fixture_html):
    with pytest.raises(ValueError, match="not present on this page"):
        parse_product_page(fixture_html("product_simple.html"), sku="999999")
