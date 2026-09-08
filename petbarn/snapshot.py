"""Loads the committed snapshot into memory.

Read once at start-up and kept. The whole dataset is a few megabytes
uncompressed, so there is nothing to gain from a database or from loading
reviews on demand, and plain objects in a list are far easier to reason about
than either.
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from pathlib import Path

from petbarn.models import SCHEMA_VERSION, Catalogue, Manifest, Product, Review

DEFAULT_DIR = Path(__file__).resolve().parent.parent / "data" / "snapshot"


class SnapshotError(RuntimeError):
    """The snapshot is missing, unreadable, or not the version we expect."""


@dataclass(frozen=True)
class Snapshot:
    """Everything the app knows, with the reviews already grouped by product."""

    captured_at: datetime
    products: dict[str, Product]
    reviews: dict[str, list[Review]]
    manifest: Manifest

    @property
    def product_list(self) -> list[Product]:
        return list(self.products.values())

    @property
    def total_reviews(self) -> int:
        return sum(len(r) for r in self.reviews.values())

    def age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.captured_at.date()).days


def load(directory: Path | None = None) -> Snapshot:
    directory = directory or DEFAULT_DIR

    catalog_file = directory / "catalog.json"
    reviews_file = directory / "reviews.jsonl.gz"
    manifest_file = directory / "manifest.json"

    missing = [f.name for f in (catalog_file, reviews_file, manifest_file) if not f.exists()]
    if missing:
        raise SnapshotError(
            f"snapshot incomplete in {directory}: missing {', '.join(missing)}. "
            "Run python -m petbarn.ingest to build it."
        )

    catalogue = Catalogue.model_validate_json(catalog_file.read_text(encoding="utf-8"))
    manifest = Manifest.model_validate_json(manifest_file.read_text(encoding="utf-8"))

    # A snapshot written by an older ingest may use a field to mean something
    # different. Refusing it is better than reading it under the wrong
    # assumptions, which would be invisible.
    if catalogue.schema_version != SCHEMA_VERSION:
        raise SnapshotError(
            f"snapshot is schema {catalogue.schema_version}, this code expects "
            f"{SCHEMA_VERSION}. Re-run the ingest."
        )

    reviews: dict[str, list[Review]] = {p.catalogue_id: [] for p in catalogue.products}
    with gzip.open(reviews_file, "rt", encoding="utf-8") as fh:
        for line in fh:
            review = Review.model_validate_json(line)
            reviews.setdefault(review.catalogue_id, []).append(review)

    orphans = set(reviews) - {p.catalogue_id for p in catalogue.products}
    if orphans:
        raise SnapshotError(f"reviews reference unknown products: {sorted(orphans)}")

    return Snapshot(
        captured_at=catalogue.captured_at,
        products={p.catalogue_id: p for p in catalogue.products},
        reviews=reviews,
        manifest=manifest,
    )


@lru_cache(maxsize=1)
def get() -> Snapshot:
    """The shared snapshot. Cached because nothing about it changes at runtime."""
    return load()
