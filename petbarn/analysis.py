"""Statistics and quote selection over the review corpus.

Everything here is a pure function over data already on disk: no network, no
model, no randomness. That means the numbers are the same every time they are
asked for, they can be checked by hand, and they can be tested without an API
key.

The division this file exists to enforce: it does the counting, and the
language model does the reading. A model asked to count reviews will guess, and
the guess will be plausible.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

from petbarn import aspects
from petbarn.models import Product, Review

RECENT_WINDOW_DAYS = 90

# Below this, one review moves the mean by more than a star and the delta says
# more about who happened to write than about the product.
MIN_MENTIONS_FOR_CONFIDENCE = 5

# Reviewers who are broadly happy still say what annoyed them, and in a corpus
# that is 96% four and five stars this is where nearly all the criticism lives.
# One written complaint exists for the Royal Canin; mining the concessions in
# its praise is the only honest way to answer a question about drawbacks.
# Markers that are concessive on their own. If someone writes "my only
# complaint" or "would be better if", what follows is a reservation whatever
# else is in the sentence.
STRONG_MARKER = re.compile(
    r"""\b(
        only\s+(downside|complaint|problem|issue|negative|gripe|criticism)
      | my\s+only\b | downside\s+(is|was) | shame\s+(that|is)
      | wish\s+(it|they|there|the) | would\s+be\s+better\s+if
      | if\s+only | let\s+down\s+by
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# These often introduce praise instead: "expensive but worth it", "the only
# brand he will eat". They only count when the clause that follows actually
# complains about something, which the hints below test for.
WEAK_MARKER = re.compile(
    r"\b(but|however|although|though|apart\s+from|other\s+than\s+that)\b",
    re.IGNORECASE,
)

# Substantive complaint vocabulary only, and it has been narrowed twice.
# Bare negations matched the opposite of a complaint far more often than a
# complaint: "no issues finishing it", "not one upset kitty". Words describing
# the animal rather than the product did the same: "my fussy cat loved it" and
# "my dog prefers it" are recommendations, and between them fussy, picky and
# prefer produced twenty-two of Pro Plan's twenty-three caveats, all praise.
COMPLAINT_HINT = re.compile(
    r"""\b(
        difficult | expensive | pricey | costly | overpriced | price | cost
      | increase | increased | went\s+up | dust | dusty | smell | smelly
      | odour | odor | mess | messy | stale | soggy | crumbl | brittle
      | refuse | refused | sick | vomit | diarrh | itch | disappoint
      | annoying | frustrat | wrong | missing | faulty | leak | tear | tore
      | ripped | shortage | discontinued | wish | unfortunately | downside
      | drawback | shame | hassle | struggle | dislike | issues? | problems?
      | complaints? | too\s+(big|small|hard|soft)
    )""",
    re.IGNORECASE | re.VERBOSE,
)

# "no issues", "never had a problem": the complaint word is present and the
# meaning is the opposite. Vetoes the clause outright.
NEGATED_COMPLAINT = re.compile(
    r"\b(no|not|n't|never|without|zero)\s+(\w+\s+){0,2}"
    r"(issues?|problems?|complaints?|trouble|side\s+effects?|downsides?)\b",
    re.IGNORECASE,
)

SENTENCE_BOUNDARY = re.compile(r"[.!?\n]")


@dataclass(frozen=True)
class AspectStat:
    aspect: str
    label: str
    mentions: int
    share_of_reviews: float
    mean_rating: float
    # Mentioners' mean minus the mean of everyone who did not mention it. The
    # comparison is against the others, not against the whole corpus: including
    # the mentioners in their own baseline flattens the difference in
    # proportion to how many of them there are, which was up to threefold here.
    delta_vs_others: float
    low_confidence: bool


@dataclass(frozen=True)
class Quote:
    review_id: str
    rating: int
    title: str | None
    body: str
    submitted_on: date
    verified_purchaser: bool
    helpful_up: int


@dataclass(frozen=True)
class Caveat:
    review_id: str
    rating: int
    submitted_on: date
    clause: str


def mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 2) if values else 0.0


def matching(reviews: list[Review], pattern: re.Pattern[str]) -> list[Review]:
    return [
        r for r in reviews
        if pattern.search(r.body) or (r.title and pattern.search(r.title))
    ]


def aspect_row(
    reviews: list[Review], pattern: re.Pattern[str], *, aspect: str, label: str
) -> AspectStat | None:
    """One row of the aspect table. None when nobody raised the subject."""
    hits = matching(reviews, pattern)
    if not hits:
        return None

    hit_ids = {r.review_id for r in hits}
    others = [r for r in reviews if r.review_id not in hit_ids]

    hit_mean = sum(r.rating for r in hits) / len(hits)
    other_mean = sum(r.rating for r in others) / len(others) if others else hit_mean

    return AspectStat(
        aspect=aspect,
        label=label,
        mentions=len(hits),
        share_of_reviews=round(len(hits) / len(reviews), 3),
        mean_rating=round(hit_mean, 2),
        # Rounded once, at the end, so the published delta is the rounded
        # difference rather than a difference of roundings.
        delta_vs_others=round(hit_mean - other_mean, 2),
        low_confidence=len(hits) < MIN_MENTIONS_FOR_CONFIDENCE or not others,
    )


def aspect_table(product: Product, reviews: list[Review]) -> list[AspectStat]:
    """Mentions and rating delta for each aspect, busiest first.

    delta_vs_others is the number that carries meaning. A negative delta says
    the people who raised this subject rated the product lower than the people
    who did not, which is a fact about their own scores rather than a reading
    of their words.

    It also survives the obvious weakness of keyword matching, but only where
    the sample is large enough to. "No side effects" matches the same terms as
    a complaint about side effects, and where enough people write it the delta
    comes out positive. Where three people wrote it, one dissenter moves the
    mean by more than a star, so those rows are marked low_confidence rather
    than trusted.
    """
    if not reviews:
        return []

    rows = [
        aspect_row(reviews, pattern, aspect=name, label=aspects.READABLE.get(name, name))
        for name, pattern in aspects.for_product(product.category, product.display_name).items()
    ]
    return sorted((r for r in rows if r), key=lambda a: a.mentions, reverse=True)


def recent(reviews: list[Review], *, today: date | None = None,
           days: int = RECENT_WINDOW_DAYS) -> list[Review]:
    cutoff = (today or date.today()) - timedelta(days=days)
    return [r for r in reviews if r.submitted_on >= cutoff]


def _sentence_around(text: str, index: int) -> str:
    """The whole sentence containing a position, not the tail after it.

    Taking the tail was the original approach and it broke every clause it
    produced: "The only downside to black hank is the cost" came back as "to
    black hank is the cost", and "just wish the price had not increased so
    much" came back asserting the opposite of the complaint.
    """
    left = 0
    for boundary in SENTENCE_BOUNDARY.finditer(text, 0, index):
        left = boundary.end()

    end = SENTENCE_BOUNDARY.search(text, index)
    right = end.start() if end else len(text)

    return text[left:right].strip(" ,:;-\t")


def harvest_caveats(reviews: list[Review], *, limit: int = 8) -> list[Caveat]:
    """Reservations expressed inside otherwise positive reviews.

    Criticism is four percent of this corpus and one product has a single
    written complaint, so the drawbacks people mention in passing while
    recommending something are most of what there is to find.

    Two kinds of marker, because they behave differently. "My only complaint"
    introduces a reservation whatever follows it. "But" does not: the complaint
    in "expensive but worth every cent" sits to the left of the marker and the
    praise to the right, so a weak marker only counts when the text it
    introduces actually complains about something.

    Either way the whole sentence is returned rather than a fragment of it. The
    model needs to be able to quote this to a person.
    """
    found: list[Caveat] = []

    for review in reviews:
        if review.rating < 4:
            continue

        strong = STRONG_MARKER.search(review.body)
        match = strong or WEAK_MARKER.search(review.body)
        if not match:
            continue

        clause = _sentence_around(review.body, match.start())

        if not strong:
            # Only the words between the marker and the end of ITS sentence
            # decide this. Searching the rest of the review admits a clause
            # because something unrelated three sentences later complains.
            end = SENTENCE_BOUNDARY.search(review.body, match.end())
            rest = review.body[match.end(): end.start() if end else len(review.body)]
            if not COMPLAINT_HINT.search(rest) or NEGATED_COMPLAINT.search(rest):
                continue

        # A fragment reads as noise; an over-long one has wandered off subject.
        if not 20 <= len(clause) <= 220 or len(clause.split()) < 5:
            continue

        found.append(
            Caveat(
                review_id=review.review_id,
                rating=review.rating,
                submitted_on=review.submitted_on,
                clause=clause,
            )
        )

    found.sort(key=lambda c: c.submitted_on, reverse=True)
    return found[:limit]


def _as_quote(review: Review) -> Quote:
    return Quote(
        review_id=review.review_id,
        rating=review.rating,
        title=review.title,
        body=review.body,
        submitted_on=review.submitted_on,
        verified_purchaser=review.verified_purchaser,
        helpful_up=review.helpful_up,
    )


def sample(
    reviews: list[Review], *, limit: int = 8, critical_slots: int = 3
) -> list[Quote]:
    """A deliberately unrepresentative selection, and it has to be.

    Ninety-six percent of this corpus is four or five stars, so any sample that
    mirrors the real proportions is all praise, and a question about drawbacks
    gets answered with "there don't appear to be any". Slots are reserved for
    the critical reviews and filled first.

    Critical reviews are ordered by recency because a question about drawbacks
    usually means current ones. Positive reviews are ordered by how many other
    shoppers found them helpful, which is a better signal of a useful review
    than its date.

    The caller is told what this sample is, so the model can say so rather than
    presenting it as a representative cross-section.
    """
    critical = sorted(
        (r for r in reviews if r.rating <= 3),
        key=lambda r: r.submitted_on, reverse=True,
    )[:critical_slots]

    positive = sorted(
        (r for r in reviews if r.rating >= 4),
        key=lambda r: (r.helpful_up, r.submitted_on), reverse=True,
    )

    # Clip the reserved slots against the limit too: a caller asking for two
    # quotes must not receive three because three were reserved.
    critical = critical[:limit]
    chosen = critical + positive[: max(limit - len(critical), 0)]
    return [_as_quote(r) for r in chosen]
