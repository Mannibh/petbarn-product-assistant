"""Turning what someone typed into one of nine products.

The rule under test throughout: never guess silently. Getting the wrong product
confidently is worse than asking which one was meant.
"""

from __future__ import annotations

import pytest

from petbarn.catalogue import CATALOGUE
from petbarn.resolve import resolve


@pytest.mark.parametrize("entry", CATALOGUE, ids=lambda e: e.catalogue_id)
def test_every_product_is_reachable_by_id_name_sku_and_alias(entry):
    for key in (entry.catalogue_id, entry.display_name, entry.sku, entry.slug, *entry.aliases):
        assert resolve(key).catalogue_id == entry.catalogue_id, key


def test_a_misspelling_still_resolves():
    assert resolve("royal canan maxi").catalogue_id == "royal-canin-maxi"


def test_a_category_naming_one_product_resolves():
    """We stock a single cat food, so refusing to answer "cat food" would be
    pedantic rather than careful.

    Uses "cat food" rather than "cat litter": the latter is also an alias, so it
    resolves through the index and never reaches the category rule this names."""
    assert resolve("cat food").catalogue_id == "pro-plan-chicken-cat-pouch"
    assert resolve("wet food").catalogue_id == "pro-plan-chicken-cat-pouch"


def test_a_partial_match_says_which_words_it_ignored():
    """We stock no Black Hawk puppy food, so silently answering about the adult
    food would be the confident wrong answer."""
    found = resolve("black hawk puppy food")
    assert found.catalogue_id == "black-hawk-lamb-rice"
    assert "puppy" in found.note


def test_an_exact_match_carries_no_note():
    assert resolve("the dog food from black hawk").note == ""


@pytest.mark.parametrize("junk", [5, ["x"], {"a": 1}, None, 3.5])
def test_a_non_string_query_does_not_raise(junk):
    assert resolve(junk).status == "not_found"


def test_a_category_naming_several_products_asks_which():
    found = resolve("dog food")
    assert found.status == "ambiguous"
    assert found.catalogue_id is None
    assert len(found.candidates) == 3


def test_a_named_product_beats_a_category_word_in_the_same_query():
    """"The dog food from Black Hawk" names both a shelf and a product."""
    assert resolve("the dog food from black hawk").catalogue_id == "black-hawk-lamb-rice"


def test_a_short_query_inside_a_longer_alias_does_not_win_on_its_own():
    """"dog food" appears inside the alias "maxi breed dog food" without meaning
    that product, which once made a three-way category into a two-way one."""
    assert resolve("dog food").status == "ambiguous"


def test_a_product_we_do_not_stock_is_refused_with_the_catalogue():
    found = resolve("hills science diet")
    assert found.status == "not_found"
    assert len(found.candidates) == len(CATALOGUE)


def test_nothing_at_all_is_refused():
    assert resolve("").status == "not_found"
    assert resolve(None).status == "not_found"
