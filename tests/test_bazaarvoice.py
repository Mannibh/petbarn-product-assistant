"""Paging and parsing of the review API, driven by a stub rather than the network."""

from __future__ import annotations

import json
from datetime import date

from petbarn.bazaarvoice import PAGE_SIZE, _as_date, fetch_reviews
from tests.conftest import FIXTURES


class StubFetcher:
    """Serves pages from a list, and records what was asked for."""

    def __init__(self, rows: list[dict], page_size: int = PAGE_SIZE):
        self.rows = rows
        self.page_size = page_size
        self.offsets: list[int] = []

    def get_json(self, url, *, archive_as, params=None):
        offset = params["Offset"]
        self.offsets.append(offset)
        return {
            "TotalResults": len(self.rows),
            "HasErrors": False,
            "Results": self.rows[offset: offset + self.page_size],
        }


def row(n: int) -> dict:
    return {
        "Id": f"r{n}",
        "Rating": 5,
        "Title": "Good",
        "ReviewText": f"Review number {n}, the dog is happy with it.",
        "SubmissionTime": "2026-05-01T03:00:00.000+00:00",
        "Badges": {"verifiedPurchaser": True},
        "TotalPositiveFeedbackCount": 0,
        "TotalNegativeFeedbackCount": 0,
        "IsRecommended": True,
        "SecondaryRatings": {},
    }


def collect(stub) -> list:
    return fetch_reviews(stub, "key", sku="1", catalogue_id="test-product")


def test_pages_through_a_full_corpus():
    stub = StubFetcher([row(n) for n in range(250)])
    assert len(collect(stub)) == 250
    assert stub.offsets == [0, 100, 200]


def test_a_short_page_mid_list_does_not_skip_the_rows_behind_it():
    """The bug this exists for: advancing by the requested page size rather than
    the number of rows returned silently drops everything in between."""
    stub = StubFetcher([row(n) for n in range(250)], page_size=50)
    got = collect(stub)
    assert len(got) == 50, "a short page should stop paging, not jump past rows"
    assert [r.review_id for r in got] == [f"r{n}" for n in range(50)]


def test_a_review_repeated_across_pages_is_only_kept_once():
    """Offset paging over a newest-first sort repeats a row at the boundary if
    a review is posted mid-scrape."""
    rows = [row(n) for n in range(150)]
    rows[100] = rows[99]
    stub = StubFetcher(rows)
    got = collect(stub)
    assert len({r.review_id for r in got}) == len(got)


def test_empty_corpus_terminates():
    stub = StubFetcher([])
    assert collect(stub) == []
    assert stub.offsets == [0]


def test_timestamps_become_the_retailers_local_date():
    """A third of this corpus is submitted after 14:00 UTC, which is already the
    next day in Sydney, and an Australian reader takes these as local dates."""
    assert _as_date("2026-09-04T21:41:58.000+00:00") == date(2026, 9, 5)
    assert _as_date("2026-09-04T03:10:00.000+00:00") == date(2026, 9, 4)
    assert _as_date(None) is None


def test_real_captured_page_parses():
    payload = json.loads((FIXTURES / "bv_reviews_page.json").read_text())
    stub = StubFetcher(payload["Results"])
    got = collect(stub)
    assert len(got) == 3
    assert all(1 <= r.rating <= 5 for r in got)
    assert all(r.body for r in got)
