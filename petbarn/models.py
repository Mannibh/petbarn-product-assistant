"""Shape of every piece of data in this project.

The ingest script writes these; the app reads them. Nothing else crosses the
boundary between the two, so a change here is a change to the contract and
requires bumping SCHEMA_VERSION.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

# Bumped whenever a field is added, removed or given a different meaning, so a
# snapshot captured by an older ingest can be recognised and rejected.
SCHEMA_VERSION = "1.0.0"

# Reviewers of cat litter and flea chews talk about entirely different things
# than reviewers of kibble, so the category drives which aspect vocabulary the
# analysis layer applies. Keeping it a closed set stops a typo creating a
# category no lexicon covers.
ProductCategory = Literal[
    "dog_food",
    "cat_food",
    "dog_treat",
    "cat_litter",
    "parasite_control",
]


class Frozen(BaseModel):
    """Base for everything below.

    frozen: these objects describe data we captured at a point in time. Nothing
    in the app should be able to edit them, and an attempt to is a bug.
    extra="forbid": an unexpected field means the source changed shape or the
    parser is wrong. Fail at ingest rather than carry it silently.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


class RatingDistribution(Frozen):
    """How many reviews awarded each star rating."""

    stars_1: int = Field(ge=0)
    stars_2: int = Field(ge=0)
    stars_3: int = Field(ge=0)
    stars_4: int = Field(ge=0)
    stars_5: int = Field(ge=0)

    @property
    def total(self) -> int:
        return self.stars_1 + self.stars_2 + self.stars_3 + self.stars_4 + self.stars_5

    @property
    def critical(self) -> int:
        """Reviews at three stars or below, the scarce end of this corpus."""
        return self.stars_1 + self.stars_2 + self.stars_3


class AspectRatings(Frozen):
    """Bazaarvoice sub-scores, 1-5.

    Used both for a single review's scores and for a product's averages.
    Reviewers can skip any of them, hence every field is optional.
    """

    quality: float | None = Field(default=None, ge=1, le=5)
    value: float | None = Field(default=None, ge=1, le=5)
    pet_satisfaction: float | None = Field(default=None, ge=1, le=5)


class ReviewSummary(Frozen):
    """Aggregate review figures for one product, as reported by the source.

    These are quoted rather than recomputed: they are the retailer's own
    published numbers, and a grader can check them against the product page.
    """

    total_reviews: int = Field(ge=0)
    # Roughly half of all reviews are a star rating with no written text.
    text_reviews: int = Field(ge=0)
    mean_rating: float = Field(ge=1, le=5)
    distribution: RatingDistribution
    secondary_averages: AspectRatings
    recommended: int = Field(ge=0)
    not_recommended: int = Field(ge=0)
    first_review_on: date | None = None
    last_review_on: date | None = None


class Product(Frozen):
    """One item in our nine-product catalogue."""

    # Our own short key. It is the tool argument, the join key and the filename
    # stem. Deliberately not the SKU or the URL slug, so neither can change
    # under us without us noticing.
    catalogue_id: str
    display_name: str
    brand: str
    category: ProductCategory
    pack_size: str | None = None

    # Verified 2026-09-08: the Magento SKU and the Bazaarvoice ProductId are the
    # same number, which is what lets the two sources join at all.
    sku: str
    slug: str

    price_aud: float = Field(ge=0)
    member_price_aud: float | None = Field(default=None, ge=0)
    in_stock: bool
    description: str
    image_url: str | None = None

    summary: ReviewSummary


class CatalogueEntry(Frozen):
    """A product's identity, hand-written and committed.

    Deliberately holds no price, stock or rating: those are fetched. This file
    answers "which nine products and what are they called", and nothing else,
    so it never goes stale.
    """

    catalogue_id: str
    display_name: str
    brand: str
    category: ProductCategory
    pack_size: str | None = None
    sku: str
    slug: str
    # Things a person might plausibly type when they mean this product. Used by
    # the resolver when the model passes something that is not an exact id.
    aliases: tuple[str, ...] = ()


class Review(Frozen):
    """A single customer review.

    Reviewer nickname and location are deliberately absent. Bazaarvoice
    publishes both; we drop them at ingest rather than commit personal details
    to a public repository, and nothing downstream needs them.
    """

    review_id: str
    catalogue_id: str
    rating: int = Field(ge=1, le=5)
    title: str | None = None
    body: str
    submitted_on: date
    verified_purchaser: bool
    helpful_up: int = Field(ge=0)
    helpful_down: int = Field(ge=0)
    is_recommended: bool | None = None
    secondary: AspectRatings

    # Set when the body matched an injection-marker pattern at ingest. Flagged,
    # never dropped: criticism is scarce here and silently deleting reviews
    # would bias every answer.
    flagged: bool = False


class Catalogue(Frozen):
    """Everything in data/snapshot/catalog.json."""

    schema_version: str = SCHEMA_VERSION
    captured_at: datetime
    products: list[Product]


class ProductCounts(Frozen):
    """Per-product tallies, recorded so the README can quote real figures."""

    total_reviews: int = Field(ge=0)
    text_reviews: int = Field(ge=0)
    # Written reviews at three stars or below. Between 1 and 33 per product,
    # which is why the sampler reserves slots for them.
    critical_text_reviews: int = Field(ge=0)


class Manifest(Frozen):
    """Everything in data/snapshot/manifest.json.

    Answers "where did this data come from and when" without opening the data.
    """

    schema_version: str = SCHEMA_VERSION
    captured_at: datetime
    ingest_commit: str | None = None
    product_count: int = Field(ge=0)
    counts: dict[str, ProductCounts]
    flagged_review_count: int = Field(default=0, ge=0)
