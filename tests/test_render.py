"""Anything shown on the page has passed through text written by strangers."""

from __future__ import annotations

from petbarn.render import safe_markdown


def test_an_image_reference_is_removed():
    """The one that matters: a browser fetches this the moment the answer
    renders, from whichever server the review author chose.

    Uses an allowed host on purpose. With an attacker host the link rule strips
    it anyway, so the original version of this test passed with the image rule
    deleted entirely and the headline guard had no coverage at all."""
    out = safe_markdown("Customers like it ![](https://www.petbarn.com.au/x.png) a lot")
    assert "petbarn.com.au/x.png" not in out
    assert "Customers like it" in out and "a lot" in out

    # the same URL as an ordinary link is allowed through, which is what makes
    # the assertion above about images rather than about hosts
    assert "petbarn.com.au/x.png" in safe_markdown("[shot](https://www.petbarn.com.au/x.png)")


def test_every_way_an_image_can_be_written_is_defused():
    """CommonMark writes an image four ways and a pattern that enumerates them
    will always miss one. The scheme-less form survived the first version."""
    for hostile in ("![](http://evil.example/p.png)",
                    "![alt](//evil.example/p.png)",
                    "![a](<http://evil.example/p.png>)",
                    "![ref][1]"):
        out = safe_markdown(f"Nice {hostile} product")
        assert "evil.example" not in out
        assert "![" not in out, f"an image construct survived: {out!r}"


def test_an_exclamation_mark_in_ordinary_prose_is_untouched():
    text = "Rated 5 out of 5! Customers love it."
    assert safe_markdown(text) == text


def test_a_link_to_the_retailer_survives():
    out = safe_markdown("See [the product](https://www.petbarn.com.au/p/x).")
    assert out == "See [the product](https://www.petbarn.com.au/p/x)."


def test_a_link_anywhere_else_becomes_plain_text():
    out = safe_markdown("Try [this instead](https://elsewhere.example/deal).")
    assert "elsewhere.example" not in out
    assert "this instead" in out


def test_a_lookalike_host_is_not_treated_as_the_retailer():
    for url in ("https://petbarn.com.au.evil.example/x",
                "https://notpetbarn.com.au/x",
                "https://evil.example/?petbarn.com.au"):
        assert url not in safe_markdown(f"[click]({url})")


def test_a_subdomain_of_the_retailer_is_allowed():
    out = safe_markdown("[api](https://mesh.petbarn.com.au/api/graphql)")
    assert "mesh.petbarn.com.au" in out


def test_a_bare_url_elsewhere_is_removed():
    assert "evil.example" not in safe_markdown("Go to https://evil.example/now please")


def test_a_scheme_less_or_upper_case_host_is_also_removed():
    """Streamlit autolinks a bare www. host, and matching was case-sensitive.

    Compared case-insensitively: asserting the lower-case spelling is absent
    passes on an upper-case survivor without testing anything."""
    assert "evil.example" not in safe_markdown("Go to www.evil.example/now").lower()
    assert "evil.example" not in safe_markdown("Go to HTTPS://EVIL.EXAMPLE/now").lower()


def test_a_link_reference_definition_is_dropped():
    """A surviving [ref][1] resolves against a definition further down the text.

    Uses an allowed host deliberately: with an attacker host the bare-URL rule
    removes it anyway, and this test would pass with the definition rule gone."""
    out = safe_markdown("See [more][1] below\n\n[1]: https://www.petbarn.com.au/x")
    assert "[1]:" not in out
    assert "See" in out


def test_ordinary_prose_is_untouched():
    text = "Customers rate it 4.79 from 1,253 reviews; 96% are positive."
    assert safe_markdown(text) == text


def test_empty_input_is_handled():
    assert safe_markdown("") == ""
    assert safe_markdown(None) == ""
