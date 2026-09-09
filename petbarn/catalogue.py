"""The nine products this assistant knows about.

Hand-written and committed rather than discovered at runtime. The brief asks
for eight to ten products, so the closed set is the requirement, not a
shortcut: a question about anything else gets a scoped refusal that names what
is available.

Every SKU below was confirmed to resolve on both petbarn.com.au and the
Bazaarvoice review API on 2026-09-08. The two systems use the same number,
which is what lets product details and reviews join at all.

Chosen for spread across categories and for review depth: the smallest has 254
reviews, the largest 1,253. Depth matters because criticism is scarce here, and
a product with fifty reviews would have no critical reviews to draw on.
"""

from __future__ import annotations

from petbarn.models import CatalogueEntry

CATALOGUE: tuple[CatalogueEntry, ...] = (
    CatalogueEntry(
        catalogue_id="royal-canin-maxi",
        display_name="Royal Canin Maxi Breed Adult Dog Food",
        brand="Royal Canin",
        category="dog_food",
        pack_size="15kg",
        sku="30299",
        slug="royal-canin-maxi-adult-dog-food-15kg",
        aliases=(
            "royal canin maxi",
            "royal canin",
            "royal canin adult",
            "royal canin large breed",
            "royal canin for big dogs",
            "maxi breed dog food",
            "royal canin 15kg",
        ),
    ),
    CatalogueEntry(
        catalogue_id="black-hawk-lamb-rice",
        display_name="Black Hawk Lamb and Rice Adult Dog Food",
        brand="Black Hawk",
        category="dog_food",
        pack_size="20kg",
        sku="127959",
        slug="black-hawk-lamb-rice-adult-dog-food",
        aliases=(
            "black hawk lamb",
            "black hawk",
            "black hawk lamb and rice",
            "blackhawk lamb rice",
            "black hawk adult dog food",
            "black hawk 20kg",
        ),
    ),
    CatalogueEntry(
        catalogue_id="prime100-roo-roll",
        display_name="Prime100 Kangaroo and Pumpkin Cooked Dog Roll",
        brand="Prime100",
        category="dog_food",
        pack_size="2kg",
        sku="133076",
        slug="prime100-kangaroo-pumpkin-cooked-dog-roll-2kg",
        aliases=(
            "prime100",
            "prime 100",
            "prime 100 roll",
            "kangaroo dog roll",
            "roo and pumpkin",
            "prime100 kangaroo",
            "fresh dog roll",
        ),
    ),
    CatalogueEntry(
        catalogue_id="pro-plan-chicken-cat-pouch",
        display_name="Pro Plan Chicken Adult Cat Food Pouches",
        brand="Pro Plan",
        category="cat_food",
        pack_size="85g x 12",
        sku="138484",
        slug="pro-plan-chicken-adult-cat-pouch-85gx12",
        aliases=(
            "pro plan chicken",
            "pro plan",
            "pro plan cat",
            "proplan cat pouches",
            "chicken cat pouches",
            "pro plan wet cat food",
        ),
    ),
    CatalogueEntry(
        catalogue_id="breeders-choice-litter",
        display_name="Breeders Choice Cat Litter",
        brand="Breeders Choice",
        category="cat_litter",
        pack_size="30L",
        sku="117874",
        slug="breeders-choice-cat-litter",
        aliases=(
            "breeders choice",
            "breeders",
            "breeder's choice litter",
            "paper cat litter",
            "recycled paper litter",
            "cat litter",
        ),
    ),
    CatalogueEntry(
        catalogue_id="barkers-best-biscuits",
        display_name="Barkers Best Variety Bone Biscuit Dog Treats",
        brand="Barkers Best",
        category="dog_treat",
        pack_size="750g",
        sku="138869",
        slug="barkers-best-variety-bone-biscuit-dog-treat-750g",
        aliases=(
            "barkers best",
            "barkers",
            "bone biscuits",
            "barkers biscuits",
            "variety bone biscuit",
            "dog biscuits",
        ),
    ),
    CatalogueEntry(
        catalogue_id="love-em-beef-liver",
        display_name="Love Em Air Dried Beef Liver Dog Treats",
        brand="Love Em",
        category="dog_treat",
        pack_size="500g",
        sku="141648",
        slug="love-em-air-dried-beef-liver-dog-treat",
        aliases=(
            "love em",
            "love'em liver",
            "beef liver treats",
            "air dried liver",
            "liver training treats",
        ),
    ),
    CatalogueEntry(
        catalogue_id="nexgard-spectra-15-30",
        display_name="NexGard Spectra Chews for Dogs 15.1-30kg",
        brand="NexGard",
        category="parasite_control",
        pack_size="15.1-30kg",
        sku="134736",
        slug="nexgard-spectra-for-dogs-15-1-30kg",
        aliases=(
            "nexgard",
            "nexgard spectra chew",
            "nexgard spectra",
            "nexgard for large dogs",
            "flea and worm chew",
            "nexgard 15-30kg",
        ),
    ),
    CatalogueEntry(
        catalogue_id="simparica-trio-20-40",
        display_name="Simparica Trio Flea Tick and Worm Chew for Dogs 20.1-40kg",
        brand="Simparica",
        category="parasite_control",
        pack_size="20.1-40kg",
        sku="139850",
        slug="simparica-trio-20-1-40kg-dog-flea-tick-worm-chew",
        aliases=(
            "simparica",
            "simparica chew",
            "simparica trio",
            "flea tick and worm chew",
            "simparica for big dogs",
            "simparica 20-40kg",
        ),
    ),
)

BY_ID: dict[str, CatalogueEntry] = {entry.catalogue_id: entry for entry in CATALOGUE}

# The tool argument is constrained to these values, so the model cannot invent a
# product that does not exist.
CATALOGUE_IDS: tuple[str, ...] = tuple(BY_ID)
