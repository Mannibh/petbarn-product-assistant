"""Fetches everything and writes the snapshot the app reads.

Run it with:  python -m petbarn.ingest

This is the only part of the project that touches Petbarn. The chat app never
does: it reads the files this produces. That split is deliberate. Scraping is
slow, occasionally fails, and depends on a site we do not control, so it
happens here, once, under supervision, rather than while somebody waits for an
answer.

Two rules make the output trustworthy:

  Nothing partial ships. Every product is checked before anything is written,
  and a failure aborts the whole run. A snapshot where two products have
  reviews and seven do not is worse than no snapshot, because the app cannot
  tell the difference.

  Nothing is written in place. Files are built in a staging directory and moved
  over the old ones at the end, with the manifest moved last. An interrupted run
  leaves the previous snapshot in place; a crash during the final moves can only
  leave a manifest older than the data beside it, which is detectable, rather
  than a truncated file, which is not.
"""

from __future__ import annotations

import argparse
import gzip
import logging
import os
import shutil
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

from petbarn.bazaarvoice import discover_passkey, fetch_reviews, fetch_summaries
from petbarn.catalogue import CATALOGUE
from petbarn.fetch import Blocked, Fetcher
from petbarn.models import (
    SCHEMA_VERSION,
    Catalogue,
    CatalogueEntry,
    Manifest,
    Product,
    ProductCounts,
    Review,
    ReviewSummary,
)
from petbarn.site import fetch_product

log = logging.getLogger("ingest")

DATA_DIR = Path("data")
SNAPSHOT_DIR = DATA_DIR / "snapshot"
# A subset run must not overwrite the committed snapshot with a partial one.
PARTIAL_DIR = DATA_DIR / "snapshot-partial"
RAW_DIR = DATA_DIR / "raw"

# A product with fewer written reviews than this cannot support a useful
# answer about what customers think, so it is a hard failure rather than a
# warning. The smallest product in the catalogue has 127.
MIN_TEXT_REVIEWS = 50

# Written criticism is scarce across this whole catalogue. Below this there is
# nothing honest to say about drawbacks, which the pros-and-cons question
# needs, so it is worth a warning even though it does not stop the run.
MIN_CRITICAL_REVIEWS = 3

# Bazaarvoice occasionally reports a written review whose body is actually null,
# so the count we collect can sit a little under the count it advertises. The
# real runs show a shortfall of at most one per product; anything beyond this
# means paging stopped early and the snapshot is incomplete.
REVIEW_SHORTFALL_TOLERANCE = 5


def git_commit() -> str | None:
    """Short hash of the code that produced this snapshot, for traceability."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_product(entry: CatalogueEntry, facts: dict, summary: ReviewSummary) -> Product:
    """Combine the catalogue row, the page facts and the review aggregates."""
    description = facts.get("description") or facts.get("short_description") or ""
    return Product(
        catalogue_id=entry.catalogue_id,
        display_name=entry.display_name,
        brand=entry.brand,
        category=entry.category,
        pack_size=entry.pack_size,
        sku=entry.sku,
        slug=entry.slug,
        price_aud=facts["price_aud"],
        member_price_aud=facts.get("member_price_aud"),
        in_stock=facts["in_stock"],
        description=description,
        image_url=facts.get("image_url"),
        summary=summary,
    )


def check(
    products: list[Product], reviews: dict[str, list[Review]], expected: int
) -> list[str]:
    """Return every reason this snapshot should not ship. Empty means good.

    expected is how many products this run asked for, not how many exist, so a
    deliberate subset run can still be validated on its own terms.
    """
    problems: list[str] = []

    if len(products) != expected:
        problems.append(f"expected {expected} products, built {len(products)}")

    ids = [r.review_id for group in reviews.values() for r in group]
    if len(ids) != len(set(ids)):
        problems.append(f"{len(ids) - len(set(ids))} duplicate review ids")

    for product in products:
        found = reviews.get(product.catalogue_id, [])
        where = product.catalogue_id

        if not product.price_aud:
            problems.append(f"{where}: no price")
        if not product.description:
            problems.append(f"{where}: no description")
        if len(found) < MIN_TEXT_REVIEWS:
            problems.append(
                f"{where}: only {len(found)} written reviews, need {MIN_TEXT_REVIEWS}"
            )

        # The source tells us how many written reviews exist. Collecting far
        # fewer means paging stopped early, which is otherwise invisible: the
        # snapshot looks complete and simply has less to say.
        shortfall = product.summary.text_reviews - len(found)
        if shortfall > REVIEW_SHORTFALL_TOLERANCE:
            problems.append(
                f"{where}: collected {len(found)} written reviews but the source "
                f"reports {product.summary.text_reviews}"
            )

        distribution = product.summary.distribution
        if distribution.total != product.summary.total_reviews:
            problems.append(
                f"{where}: star counts total {distribution.total} but the summary "
                f"says {product.summary.total_reviews}"
            )

        critical = sum(1 for r in found if r.rating <= 3)
        if critical < MIN_CRITICAL_REVIEWS:
            log.warning(
                "%s: only %d written reviews at 3 stars or below; "
                "answers about drawbacks will be thin",
                where, critical,
            )

    return problems


def write_snapshot(
    catalogue: Catalogue, reviews: list[Review], manifest: Manifest, out_dir: Path
) -> None:
    """Build the three files elsewhere, then move them into place."""
    staging = out_dir.parent / f".{out_dir.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    (staging / "catalog.json").write_text(
        catalogue.model_dump_json(indent=2), encoding="utf-8"
    )

    # One review per line rather than one big array: the file can be read a
    # line at a time, and a diff between two snapshots shows the reviews that
    # changed instead of reformatting the whole file.
    with gzip.open(staging / "reviews.jsonl.gz", "wt", encoding="utf-8") as fh:
        for review in reviews:
            fh.write(review.model_dump_json() + "\n")

    (staging / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )

    # The manifest goes last and acts as the marker that the rest is complete,
    # since three separate moves cannot be made a single atomic operation.
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("catalog.json", "reviews.jsonl.gz", "manifest.json"):
        os.replace(staging / name, out_dir / name)
    shutil.rmtree(staging)


def run(only: list[str] | None = None, delay: float = 1.5) -> int:
    wanted = [e for e in CATALOGUE if not only or e.catalogue_id in only]
    raw_dir = RAW_DIR / date.today().isoformat()

    products: list[Product] = []
    reviews_by_product: dict[str, list[Review]] = {}

    try:
        with Fetcher(raw_dir, delay=delay) as fetcher:
            passkey = discover_passkey(fetcher)
            log.info("display key resolved from Bazaarvoice config")

            summaries = fetch_summaries(fetcher, passkey, [e.sku for e in wanted])
            missing = [e.catalogue_id for e in wanted if e.sku not in summaries]
            if missing:
                log.error("no review statistics returned for: %s", ", ".join(missing))
                return 1

            for position, entry in enumerate(wanted, start=1):
                log.info("[%d/%d] %s", position, len(wanted), entry.catalogue_id)

                facts = fetch_product(
                    fetcher, slug=entry.slug, sku=entry.sku,
                    catalogue_id=entry.catalogue_id,
                )
                products.append(build_product(entry, facts, summaries[entry.sku]))
                reviews_by_product[entry.catalogue_id] = fetch_reviews(
                    fetcher, passkey, sku=entry.sku, catalogue_id=entry.catalogue_id
                )

            request_count = fetcher.request_count

    except Blocked as exc:
        log.error("stopped: %s", exc)
        return 2

    problems = check(products, reviews_by_product, expected=len(wanted))
    if problems:
        log.error("snapshot rejected, nothing written:")
        for problem in problems:
            log.error("  %s", problem)
        return 1

    all_reviews = [r for group in reviews_by_product.values() for r in group]
    captured_at = datetime.now(timezone.utc)
    out_dir = PARTIAL_DIR if only else SNAPSHOT_DIR
    if only:
        log.warning(
            "subset run: writing to %s so the committed snapshot is left alone", out_dir
        )

    write_snapshot(
        Catalogue(schema_version=SCHEMA_VERSION, captured_at=captured_at, products=products),
        all_reviews,
        Manifest(
            schema_version=SCHEMA_VERSION,
            captured_at=captured_at,
            ingest_commit=git_commit(),
            product_count=len(products),
            counts={
                p.catalogue_id: ProductCounts(
                    total_reviews=p.summary.total_reviews,
                    text_reviews=len(reviews_by_product[p.catalogue_id]),
                    critical_text_reviews=sum(
                        1 for r in reviews_by_product[p.catalogue_id] if r.rating <= 3
                    ),
                )
                for p in products
            },
            flagged_review_count=sum(1 for r in all_reviews if r.flagged),
        ),
        out_dir,
    )

    log.info(
        "wrote %d products and %d reviews to %s in %d requests; raw responses archived to %s",
        len(products), len(all_reviews), out_dir, request_count, raw_dir,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only", nargs="*", metavar="ID",
        help="catalogue ids to fetch instead of all nine, for quicker testing; "
             "writes to data/snapshot-partial rather than the committed snapshot",
    )
    parser.add_argument(
        "--delay", type=float, default=1.5,
        help="seconds between requests (default: 1.5)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    return run(only=args.only, delay=args.delay)


if __name__ == "__main__":
    sys.exit(main())
