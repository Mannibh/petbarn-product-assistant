"""Reads customer reviews from Bazaarvoice, the service Petbarn uses.

Petbarn's product pages contain no review text at all. The reviews tab is an
empty placeholder that the browser fills in afterwards by calling Bazaarvoice
directly. So an HTML scraper finds nothing here, and the reviews have to come
from the same public API the browser uses.

Bazaarvoice keys reviews on the retailer's own SKU, which is why the catalogue
stores one number that works for both sources.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from zoneinfo import ZoneInfo

from petbarn.fetch import Fetcher
from petbarn.models import AspectRatings, RatingDistribution, Review, ReviewSummary
from petbarn.sanitise import clean_review_text

log = logging.getLogger(__name__)

API = "https://api.bazaarvoice.com/data"
API_VERSION = "5.4"

# Petbarn's Bazaarvoice deployment. These identifiers are in the page source of
# every product page, because the browser needs them to render the reviews.
CLIENT = "petbarn-au"
SITE = "main_site"
LOCALE = "en_AU"
CONFIG_URL = (
    f"https://apps.bazaarvoice.com/deployments/{CLIENT}/{SITE}/production/{LOCALE}"
    "/ratings-config.js"
)

# Bazaarvoice refuses a Limit above this.
PAGE_SIZE = 100

# Bazaarvoice timestamps are UTC. An Australian shopper reads a review date as a
# local date, and a third of this corpus was submitted after 14:00 UTC, which is
# already the next day in Sydney. Truncating the UTC value would date those
# reviews a day early.
RETAILER_TZ = ZoneInfo("Australia/Sydney")

_PASSKEY = re.compile(r'"mprAPIKey"\s*:\s*"([A-Za-z0-9]+)"')


def discover_passkey(fetcher: Fetcher) -> str:
    """Read the display key out of Bazaarvoice's public configuration.

    The key is not a secret: it is a display-only credential served to every
    visitor's browser so the reviews widget can load. Reading it at run time
    rather than pasting it into the source means a rotation fixes itself, and
    nothing key-shaped is committed to a public repository.
    """
    body = fetcher.get(CONFIG_URL, archive_as="bv_config.js").text
    match = _PASSKEY.search(body)
    if not match:
        raise RuntimeError(
            f"no display key found in {CONFIG_URL}; Bazaarvoice changed their config format"
        )
    return match.group(1)


def _as_date(value: str | None) -> date | None:
    """The retailer's local calendar day, which is how a reader will take it."""
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(RETAILER_TZ).date()


def _distribution(rows: list[dict]) -> RatingDistribution:
    counts = {row["RatingValue"]: row["Count"] for row in rows or []}
    return RatingDistribution(
        stars_1=counts.get(1, 0),
        stars_2=counts.get(2, 0),
        stars_3=counts.get(3, 0),
        stars_4=counts.get(4, 0),
        stars_5=counts.get(5, 0),
    )


def _secondary_averages(block: dict | None) -> AspectRatings:
    block = block or {}

    def avg(name: str) -> float | None:
        entry = block.get(name)
        return entry.get("AverageRating") if entry else None

    return AspectRatings(
        quality=avg("Quality"),
        value=avg("Value"),
        pet_satisfaction=avg("Petsatisfaction"),
    )


def _secondary_single(block: dict | None) -> AspectRatings:
    """Per-review sub-scores. Same three axes, but the value sits under a
    different key than in the aggregate block."""
    block = block or {}

    def val(name: str) -> float | None:
        entry = block.get(name)
        return entry.get("Value") if entry else None

    return AspectRatings(
        quality=val("Quality"),
        value=val("Value"),
        pet_satisfaction=val("Petsatisfaction"),
    )


def fetch_summaries(fetcher: Fetcher, passkey: str, skus: list[str]) -> dict[str, ReviewSummary]:
    """Aggregate figures for every product, in a single request.

    products.json accepts many ids at once, unlike reviews.json, so nine
    products cost one call rather than nine.
    """
    payload = fetcher.get_json(
        f"{API}/products.json",
        archive_as="bv_products.json",
        params={
            "apiversion": API_VERSION,
            "passkey": passkey,
            "Filter": f"Id:eq:{','.join(skus)}",
            "Stats": "Reviews",
        },
    )

    summaries: dict[str, ReviewSummary] = {}
    for result in payload.get("Results", []):
        stats = result.get("ReviewStatistics") or {}
        total = stats.get("TotalReviewCount", 0)
        ratings_only = stats.get("RatingsOnlyReviewCount", 0)
        summaries[str(result["Id"])] = ReviewSummary(
            total_reviews=total,
            text_reviews=max(total - ratings_only, 0),
            mean_rating=stats.get("AverageOverallRating") or 0,
            distribution=_distribution(stats.get("RatingDistribution")),
            secondary_averages=_secondary_averages(stats.get("SecondaryRatingsAverages")),
            recommended=stats.get("RecommendedCount") or 0,
            not_recommended=stats.get("NotRecommendedCount") or 0,
            first_review_on=_as_date(stats.get("FirstSubmissionTime")),
            last_review_on=_as_date(stats.get("LastSubmissionTime")),
        )
    return summaries


def fetch_reviews(
    fetcher: Fetcher, passkey: str, *, sku: str, catalogue_id: str
) -> list[Review]:
    """Every written review for one product.

    Roughly half of all reviews are a star rating with no text. Those are
    already counted in the aggregate figures and have nothing to quote, so the
    filter excludes them and the paging cost halves.
    """
    reviews: list[Review] = []
    seen: set[str] = set()
    offset = 0
    total = None

    while True:
        payload = fetcher.get_json(
            f"{API}/reviews.json",
            archive_as=f"bv_reviews_{catalogue_id}_{offset:04d}.json",
            params={
                "apiversion": API_VERSION,
                "passkey": passkey,
                "Filter": [f"ProductId:{sku}", "IsRatingsOnly:eq:false"],
                "Limit": PAGE_SIZE,
                "Offset": offset,
                "Sort": "SubmissionTime:desc",
            },
        )

        if total is None:
            total = payload.get("TotalResults", 0)
            log.info("%s: %d written reviews to collect", catalogue_id, total)

        batch = payload.get("Results", [])
        if not batch:
            break

        for row in batch:
            review_id = str(row["Id"])
            # Paging by offset over a newest-first sort shifts every row along if
            # a review is posted mid-scrape, which would hand us the same one
            # twice at a page boundary.
            if review_id in seen:
                continue
            seen.add(review_id)

            body, flagged = clean_review_text(row.get("ReviewText"))
            if not body:
                continue
            title, title_flagged = clean_review_text(row.get("Title"), max_chars=120)
            reviews.append(
                Review(
                    review_id=review_id,
                    catalogue_id=catalogue_id,
                    rating=row["Rating"],
                    title=title or None,
                    body=body,
                    submitted_on=_as_date(row["SubmissionTime"]),
                    verified_purchaser=bool((row.get("Badges") or {}).get("verifiedPurchaser")),
                    helpful_up=row.get("TotalPositiveFeedbackCount") or 0,
                    helpful_down=row.get("TotalNegativeFeedbackCount") or 0,
                    is_recommended=row.get("IsRecommended"),
                    secondary=_secondary_single(row.get("SecondaryRatings")),
                    flagged=flagged or title_flagged,
                )
            )

        # Advance by what we were actually given, not by what we asked for: a
        # short page in the middle of the list would otherwise skip every row
        # between the end of this page and the next requested offset.
        offset += len(batch)
        if len(batch) < PAGE_SIZE or offset >= total:
            break

    return reviews
