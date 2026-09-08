"""Shared test helpers.

Everything here runs offline. No network, no API key, no snapshot required, so
the suite works on a fresh clone and in CI.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from petbarn.models import AspectRatings, Review

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def fixture_html():
    def read(name: str) -> str:
        return (FIXTURES / name).read_text(encoding="utf-8")
    return read


def make_review(
    rating: int = 5,
    body: str = "Good product, the dog is happy.",
    *,
    review_id: str = "1",
    days_ago: int = 0,
    helpful: int = 0,
    title: str | None = None,
    catalogue_id: str = "test-product",
) -> Review:
    return Review(
        review_id=review_id,
        catalogue_id=catalogue_id,
        rating=rating,
        title=title,
        body=body,
        submitted_on=date(2026, 9, 8) - timedelta(days=days_ago),
        verified_purchaser=True,
        helpful_up=helpful,
        helpful_down=0,
        is_recommended=rating >= 4,
        secondary=AspectRatings(),
    )
