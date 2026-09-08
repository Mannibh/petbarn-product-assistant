"""Reads product facts from petbarn.com.au product pages.

Petbarn renders its pages on the server, so everything we need is already in
the HTML and no browser is required. Each page carries schema.org JSON-LD, the
same structured block search engines read, which is far more stable to parse
than the page's visible markup.

The catch is that the block comes in three shapes. See collect_products.
"""

from __future__ import annotations

import json
import logging
import re

from petbarn.fetch import Blocked, Fetcher
from petbarn.sanitise import CONTROLS, HTML_TAG

log = logging.getLogger(__name__)

PRODUCT_URL = "https://www.petbarn.com.au/p/{slug}"
GRAPHQL_URL = "https://www.petbarn.com.au/graphql"

LD_BLOCK = re.compile(
    r'<script[^>]+type="application/ld\+json"[^>]*>(.*?)</script>', re.S
)

IN_STOCK = "https://schema.org/InStock"


def collect_products(html: str) -> list[dict]:
    """Every product-like object on the page, flattened into one list.

    Three shapes exist and a parser that expects one silently returns nothing on
    the others:

      1. type Product        - a single item, carries sku and offers
      2. type ProductGroup   - several sizes, with a group-level rating
      3. type ProductGroup   - several sizes, with no sku and no rating at all

    Rather than branch on shape, flatten everything and let the caller pick the
    entry whose sku matches the one recorded in the catalogue. Shape three then
    needs no special handling, and a fourth shape appearing later would not
    break us.
    """
    found: list[dict] = []

    for block in LD_BLOCK.finditer(html):
        try:
            parsed = json.loads(block.group(1))
        except json.JSONDecodeError:
            continue

        for node in parsed if isinstance(parsed, list) else [parsed]:
            if not isinstance(node, dict):
                continue
            if node.get("@type") in {"Product", "ProductGroup"}:
                found.append(node)
                for variant in node.get("hasVariant") or []:
                    if isinstance(variant, dict):
                        found.append(variant)

    return found


def _offer_prices(node: dict) -> tuple[float | None, float | None, bool]:
    """Shelf price, loyalty-member price, and whether it is in stock."""
    offers = node.get("offers") or {}
    if isinstance(offers, list):
        offers = offers[0] if offers else {}

    price = offers.get("price")
    in_stock = offers.get("availability") == IN_STOCK

    # priceSpecification holds several entries: the shelf price and, when one
    # exists, the loyalty price. They share a type and are told apart only by
    # validForMemberTier, so matching on type alone silently returns the shelf
    # price as the member price.
    member = None
    for spec in offers.get("priceSpecification") or []:
        if isinstance(spec, dict) and spec.get("validForMemberTier"):
            member = spec.get("price")
            break

    return (
        float(price) if price is not None else None,
        float(member) if member is not None else None,
        in_stock,
    )


def parse_product_page(html: str, *, sku: str) -> dict:
    """Facts for one known SKU, taken from the page's structured data."""
    candidates = collect_products(html)
    match = next((n for n in candidates if str(n.get("sku")) == sku), None)

    if match is None:
        shapes = sorted({n.get("@type", "?") for n in candidates})
        raise ValueError(
            f"sku {sku} not present on this page; found {len(candidates)} "
            f"product objects of types {shapes}"
        )

    price, member_price, in_stock = _offer_prices(match)

    # A group-level rating is used when the variant has none of its own, which
    # is correct here: Bazaarvoice pools reviews across sizes of one product.
    group = next((n for n in candidates if n.get("@type") == "ProductGroup"), {})
    rating = match.get("aggregateRating") or group.get("aggregateRating") or {}

    return {
        "sku": sku,
        "name": match.get("name"),
        "brand": (match.get("brand") or group.get("brand") or {}).get("name"),
        "gtin": match.get("gtin"),
        "price_aud": price,
        "member_price_aud": member_price,
        "in_stock": in_stock,
        "image_url": match.get("image") or group.get("image"),
        "rating_value": rating.get("ratingValue"),
        "review_count": rating.get("reviewCount"),
    }


def _plain_text(markup: str | None) -> str:
    if not markup:
        return ""
    text = HTML_TAG.sub(" ", markup)
    text = CONTROLS.sub("", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


DESCRIPTION_QUERY = """
{ products(filter: {url_key: {eq: "%s"}}) { items {
    sku name url_key
    description { html }
    short_description { html }
} } }
"""


def fetch_description(fetcher: Fetcher, *, slug: str) -> dict:
    """Long-form copy from Petbarn's own catalogue service.

    The structured block on the page carries only boilerplate ("Buy X for your
    pet. Shop online with Petbarn today."), which is useless to someone asking
    what a product actually is. The store's catalogue API returns the real
    description, so the product tool has something worth reading.

    A failure here is not immediately fatal, but the run will still be rejected:
    ingest requires a description, so a missing one fails the gate at the end
    rather than at the request. The warning below is what tells you which one.
    """
    payload = fetcher.get(
        GRAPHQL_URL,
        archive_as=f"gql_{slug}.json",
        params={"query": DESCRIPTION_QUERY % slug},
    ).json()

    # GraphQL answers HTTP 200 even when the query was rejected, reporting the
    # problem only in an errors array. Without this check a malformed query
    # looks identical to a product that simply has no description.
    if payload.get("errors"):
        detail = "; ".join(e.get("message", "?") for e in payload["errors"])
        raise RuntimeError(f"graphql rejected the query for {slug}: {detail}")

    items = ((payload.get("data") or {}).get("products") or {}).get("items") or []
    if not items:
        return {}

    item = items[0]
    return {
        "sku": str(item.get("sku")) if item.get("sku") else None,
        "description": _plain_text((item.get("description") or {}).get("html")),
        "short_description": _plain_text((item.get("short_description") or {}).get("html")),
    }


def fetch_product(fetcher: Fetcher, *, slug: str, sku: str, catalogue_id: str) -> dict:
    """Everything the product tool needs about one item."""
    html = fetcher.get(
        PRODUCT_URL.format(slug=slug), archive_as=f"page_{catalogue_id}.html"
    ).text
    facts = parse_product_page(html, sku=sku)

    try:
        facts.update({k: v for k, v in fetch_description(fetcher, slug=slug).items()
                      if k != "sku" and v})
    except Blocked:
        # Being refused is the one thing that must not be tolerated here. The
        # broad handler below exists for a missing description, not for the site
        # telling us to go away, and swallowing this would keep us hammering it.
        raise
    except Exception as exc:
        log.warning("%s: description lookup failed (%s); keeping page data", catalogue_id, exc)

    return facts
