"""What customers talk about, and how to find it in their words.

Two things are counted per aspect: how many reviews mention it, and what those
reviewers' average star rating is compared with everyone else's. The second is
the useful one. If people who mention price rate a product lower than people
who do not, that is a measurement taken from the customers' own scores rather
than an opinion inferred from their prose.

Matching is keyword-based, which will miss paraphrase. That is a deliberate
trade: it is auditable, it needs no model, and it can be corrected in seconds
by editing a list. Anything cleverer is harder to defend when it is wrong.
"""

from __future__ import annotations

import re
from functools import lru_cache

from petbarn.models import ProductCategory

# Every category shares these two.
SHARED: dict[str, list[str]] = {
    "price_value": [
        "price", "prices", "priced", "pricing", "cost", "costs", "costly",
        "expensive", "cheap", "cheaper", "affordable", "value", "worth",
        "money", "bargain", "overpriced", "rip off", "dear",
    ],
    "packaging_delivery": [
        "packaging", "package", "packet", "bag", "box", "sealed", "seal",
        "delivery", "delivered", "shipping", "arrived", "postage", "resealable",
    ],
}

# Reviewers of litter and of flea chews have almost nothing in common. Giving
# each category its own vocabulary is the difference between a real breakdown
# and a table of zeroes.
BY_CATEGORY: dict[ProductCategory, dict[str, list[str]]] = {
    "dog_food": {
        "palatability": ["loves", "love it", "love them", "love these", "loved",
                          "fussy", "picky", "gobbles", "wolfs", "devours", "enjoys",
                          "refuses", "turns his nose", "turns her nose", "won't eat",
                          "wont eat", "tasty", "palatable"],
        "ingredients": ["ingredients", "protein", "grain", "grain free", "additives",
                         "preservatives", "natural", "filler", "fillers", "nutrition",
                         "nutritional", "formula", "wholesome", "by-product"],
        "digestion": ["stomach", "digest", "digestion", "digestive", "poo", "poop",
                       "stool", "stools", "diarrhoea", "diarrhea", "wind", "gas",
                       "bloating", "sensitive", "upset", "vomit"],
        "coat_condition": ["coat", "fur", "shiny", "shedding", "skin", "itching",
                            "itchy", "allergy", "allergies", "energy", "weight"],
        "portion_size": ["portion", "portions", "serving", "kibble", "biscuit size",
                          "pieces", "lasts", "last long", "quantity"],
    },
    "cat_food": {
        "palatability": ["loves", "love it", "love them", "love these", "fussy",
                          "picky", "refuses", "won't eat", "wont eat", "licks",
                          "devours", "enjoys", "tasty", "gravy", "jelly", "leaves it"],
        "ingredients": ["ingredients", "protein", "grain", "additives",
                         "preservatives", "natural", "nutrition", "formula", "by-product"],
        "digestion": ["stomach", "digest", "digestion", "vomit", "vomiting", "sick",
                       "litter tray", "stool", "sensitive", "upset"],
        "coat_condition": ["coat", "fur", "shiny", "shedding", "skin", "energy", "weight"],
        "portion_size": ["portion", "portions", "serving", "pouch", "pouches",
                          "sachet", "size", "enough"],
    },
    "dog_treat": {
        "palatability": ["loves", "love it", "love them", "love these", "loved",
                          "devours", "gobbles", "enjoys", "fussy", "picky", "refuses",
                          "won't eat", "wont eat", "tasty"],
        "size_texture": ["size", "sized", "small", "large", "hard", "crunchy", "soft",
                          "brittle", "crumbly", "break", "breaks", "snap", "chewy"],
        "ingredients": ["ingredients", "natural", "additives", "preservatives",
                         "protein", "liver", "beef", "grain", "wheat"],
        "training_use": ["training", "train", "reward", "rewards", "treat time",
                          "recall", "obedience", "puppy"],
        "mess_smell": ["smell", "smells", "smelly", "odour", "odor", "mess", "messy",
                        "crumbs", "greasy", "stains"],
    },
    "cat_litter": {
        "odour_control": ["odour", "odor", "smell", "smells", "smelly", "stink",
                           "stinks", "ammonia", "fresh", "deodorise", "deodorize"],
        "dust": ["dust", "dusty", "dust free", "dustless", "powder", "sneezing"],
        "tracking": ["track", "tracks", "tracking", "everywhere", "trail", "scatter",
                      "scattered", "mess", "floor", "carpet", "paws"],
        "clumping": ["clump", "clumps", "clumping", "scoop", "scooping", "solid",
                      "breaks apart", "sticks"],
        "absorbency": ["absorb", "absorbs", "absorbent", "absorbency", "soak",
                        "soaks", "wet", "dry", "lasts", "last long", "saturated"],
        "weight_handling": ["heavy", "weight", "lift", "lifting", "carry", "bulky",
                             "pour", "pouring", "bag"],
    },
    "parasite_control": {
        "efficacy": ["works", "worked", "working", "effective", "effectiveness",
                      "fleas", "flea", "tick", "ticks", "worm", "worms", "mites",
                      "infestation", "protection", "protected", "no more", "gone"],
        "palatability": ["chew", "chews", "loves", "takes it", "spits", "refuses",
                          "won't eat", "wont eat", "tasty", "treat", "swallow",
                          "hide it", "fussy"],
        "side_effects": ["side effect", "side effects", "reaction", "vomit",
                          "vomiting", "sick", "lethargic", "lethargy", "itchy",
                          "seizure", "unwell", "diarrhoea", "diarrhea"],
        "ease_of_use": ["easy", "simple", "monthly", "dose", "dosage", "administer",
                         "apply", "vet", "prescription", "remember"],
    },
}

READABLE: dict[str, str] = {
    "price_value": "price and value",
    "packaging_delivery": "packaging and delivery",
    "palatability": "whether the pet will eat it",
    "ingredients": "ingredients and nutrition",
    "digestion": "digestion and tolerance",
    "coat_condition": "coat, skin and condition",
    "portion_size": "portion size and how long it lasts",
    "size_texture": "size and texture",
    "training_use": "use in training",
    "mess_smell": "mess and smell",
    "odour_control": "odour control",
    "dust": "dust",
    "tracking": "tracking around the house",
    "clumping": "clumping and scooping",
    "absorbency": "absorbency and how long it lasts",
    "weight_handling": "weight and handling",
    "efficacy": "whether it works",
    "side_effects": "side effects",
    "ease_of_use": "ease of use",
}


def _pattern(terms: list[str]) -> re.Pattern[str]:
    """Word-boundary alternation, longest first so multi-word terms win.

    Boundaries matter more than they look: without them "price" matches inside
    "priceless" and "dear" inside "dearly", both of which invert the meaning.
    """
    ordered = sorted(terms, key=len, reverse=True)
    return re.compile(
        r"\b(?:" + "|".join(re.escape(t) for t in ordered) + r")\b", re.IGNORECASE
    )


@lru_cache(maxsize=None)
def for_category(category: ProductCategory) -> dict[str, re.Pattern[str]]:
    """The aspects worth counting for this kind of product."""
    terms = {**SHARED, **BY_CATEGORY[category]}
    return {name: _pattern(words) for name, words in terms.items()}


@lru_cache(maxsize=None)
def for_product(category: ProductCategory, display_name: str) -> dict[str, re.Pattern[str]]:
    """Patterns with the product's own name words removed.

    A term that appears in the product's title matches every review that simply
    names the product. "Chicken" in a lexicon of ingredient words counted 95 of
    Pro Plan Chicken's 128 ingredient mentions, including "my very fussy cat
    loves proplan chicken", which is not about ingredients at all.
    """
    own = {w for w in re.split(r"[^\w']+", display_name.lower()) if w}
    patterns: dict[str, re.Pattern[str]] = {}

    for name, words in {**SHARED, **BY_CATEGORY[category]}.items():
        kept = [w for w in words if not own.issuperset(w.lower().split())]
        if kept:
            patterns[name] = _pattern(kept)

    return patterns


# A phrase longer than this is not a topic, and the tool echoes it back, so it
# is also the cap on how much caller-supplied text can enter the payload.
MAX_AD_HOC_CHARS = 40
MAX_AD_HOC_WORDS = 4


def clean_phrase(phrase: str | None) -> str:
    """The caller's topic, reduced to something safe to echo and to match on."""
    if not isinstance(phrase, str):
        return ""
    words = [w for w in re.split(r"[^\w'-]+", phrase.lower()) if len(w) > 2]
    return " ".join(words[:MAX_AD_HOC_WORDS])[:MAX_AD_HOC_CHARS].strip()


def ad_hoc(phrase: str) -> re.Pattern[str] | None:
    """A pattern for something the user asked about that no lexicon covers.

    This is what keeps the fixed vocabulary from being a ceiling. Someone asking
    about "sensitive stomach" or "puppies" gets the same statistics computed
    over the same corpus, just matched on their words instead of ours.

    Every word must appear, not any of them. Matching "sensitive stomach" as
    either word counts every review mentioning a stomach and every review
    mentioning a sensitive anything, and reports the total as the share of
    people discussing sensitive stomachs.
    """
    words = clean_phrase(phrase).split()
    if not words:
        return None
    lookaheads = "".join(rf"(?=.*\b{re.escape(w)}\b)" for w in words)
    return re.compile(lookaheads + r".", re.IGNORECASE | re.DOTALL)
