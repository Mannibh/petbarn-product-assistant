"""The model is the contract between the scraper and the app, so these tests
check the guarantees the rest of the project is allowed to assume."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from petbarn.models import RatingDistribution, Review
from tests.conftest import make_review


def test_rating_outside_one_to_five_is_rejected():
    with pytest.raises(ValidationError):
        make_review(rating=7)


def test_unknown_field_is_rejected():
    """A misspelled field name is the bug this catches: without it the value is
    stored under the wrong name and the real one is silently absent."""
    with pytest.raises(ValidationError) as caught:
        Review(**{**make_review().model_dump(), "ratting": 5})
    assert "ratting" in str(caught.value)


def test_reviewer_identity_cannot_be_stored():
    """Nickname and location are published by Bazaarvoice and deliberately have
    no field here, so committing them to a public repo is not possible."""
    for field in ("nickname", "user_location", "author_id"):
        with pytest.raises(ValidationError):
            Review(**{**make_review().model_dump(), field: "someone"})


def test_reviews_cannot_be_edited_after_creation():
    with pytest.raises(ValidationError):
        make_review().rating = 1


def test_distribution_totals():
    d = RatingDistribution(stars_1=1, stars_2=2, stars_3=3, stars_4=4, stars_5=5)
    assert d.total == 15
    assert d.critical == 6
