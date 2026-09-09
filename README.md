# Petbarn product assistant

A chat assistant that answers questions about nine Petbarn products by calling two tools over
real product data and 3,054 real customer reviews.

**Live app:** _URL_

```
"What are people saying about the price and quality of the Black Hawk lamb and rice?"
"Compare the reviews between the NexGard Spectra and the Simparica Trio."
"What are the main pros and cons of the Breeders Choice cat litter?"
```

## Running it

```bash
git clone https://github.com/Mannibh/petbarn-product-assistant
cd petbarn-product-assistant
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt

.venv/bin/python -m pytest tests/ -q          # 167 tests, no network, no API key
```

To run the app, put a free key in `.streamlit/secrets.toml` (see `secrets.toml.example`)
and `.venv/bin/streamlit run streamlit_app.py`. To re-scrape,
`.venv/bin/python -m petbarn.ingest`.

## How it works

```
petbarn.com.au ─┐
                ├─→ ingest ─→ data/snapshot/ ─→ tools ─→ agent ─→ chat page
Bazaarvoice ────┘   (offline)   (committed)      (2)     (LLM)
```

An offline script scrapes the products and their full review history into a committed
snapshot. The chat app reads that snapshot and gives a language model two tools over it.

| Module | Job |
|---|---|
| `models.py` | the data contract between the scraper and the app |
| `catalogue.py` | the nine products, hand-written |
| `fetch.py` `bazaarvoice.py` `site.py` | acquisition |
| `sanitise.py` | cleans untrusted review text at ingest |
| `ingest.py` | orchestrates, validates, writes the snapshot atomically |
| `snapshot.py` `aspects.py` `analysis.py` | statistics, quote sampling, caveat mining |
| `resolve.py` `tools.py` | the two functions the model calls |
| `providers.py` `prompts.py` `agent.py` | the provider chain and the tool loop |
| `render.py` `streamlit_app.py` | the page, and what is safe to display on it |

## Design decisions

### The tools do the arithmetic; the model only reads

Every number the assistant states is computed in Python over the whole corpus and handed
to the model. It never counts, averages or ranks anything.

This is the decision the project turns on. A model asked how many people complained about
price will produce a plausible number, and nobody reading the answer can tell it apart from
a real one. Moving the arithmetic into the tools makes that impossible rather than unlikely,
and it has a second benefit: a small, fast model is then sufficient, because reading numbers
and writing prose is what small models are good at. The app answers in about five seconds on
a free tier.

### Reviews are read from a committed snapshot, not fetched live

The brief permits either. Three things decided it.

**The live version has less data, not more.** Bazaarvoice serves at most 100 reviews per
request, one product at a time, at 0.9-4.4 seconds each. A tool fetching during a
conversation can afford one request, which returns the hundred most recent reviews - and in
a corpus that is 96% four and five stars, those are almost all positive. It would answer
"what are the drawbacks" with a list of strengths. Paging the whole history once, offline,
lets the tool search and balance across all of it in milliseconds.

**The deployed app cannot rehearse its own network path.** A fallback is by construction the
least-exercised code in a system, so making it the path a visitor lands on means grading day
is its first real outing.

**Sentiment needs the whole corpus.** One product's review text is roughly 55,000 tokens
against a free tier that allows 8,000 a minute. It cannot be analysed while somebody waits.

Prices are the exception: they move, and a wrong price is checkable in another browser tab
in ten seconds. Every price is shown with the date it was captured.

### Sentiment is computed, not inferred

Bazaarvoice publishes what each customer chose: a star, and separate 1-5 scores for Quality,
Value and Pet Satisfaction. Those are cited rather than guessed at. On top, `analysis.py`
computes for each topic how many reviewers raised it and how their ratings compare with
everyone else's - a measurement taken from customers' own scores, not a reading of their
prose.

No classifier is used. Running one to infer a label the reviewer already supplied would be
inventing a machine-learning problem the data does not have, and it would cost a 250MB
dependency on a host with 512MB of memory.

### Criticism is mined out of positive reviews

96% of this corpus is four or five stars. **Only 126 of 3,054 written reviews sit at three
stars or below, and one product has exactly one.** Answering "pros and cons" from those alone
would present three anecdotes as a pattern.

So reservations are harvested from otherwise positive reviews - the clause after "my only
complaint", or a "but" followed by an actual complaint. Measured by hand across the whole
corpus: 51 clauses, about 74% genuine reservations. An earlier version scored 26%, which is
why the number is stated rather than claimed.

The quote sample is also deliberately skewed towards critical reviews, and the payload says
so, so the model reports proportions from the statistics and uses quotes only as illustration.

### A provider chain, ending in no model at all

`gemini-3.5-flash-lite` → `gemini-3.6-flash` → `groq/openai/gpt-oss-120b` → the data itself.

The failure worth defending against is not a model outage. It is a mistyped key, a rate
limit, or a slow response - and to a visitor those are indistinguishable, because the host
hides error detail. So the chain moves on from almost anything, including authentication
errors, which is normally a reason to stop.

The last rung has no model in it. It resolves the question against the catalogue and renders
the underlying data with an honest banner. It cannot fail the way a model fails.

Two provider implementations exist, not one: an OpenAI-compatible client covering Gemini,
Groq and OpenRouter, and a native Anthropic client. Reaching Anthropic through its
compatibility endpoint would have made the "abstraction" a base URL.

### Scraping conduct

The ingest sends an honest, self-identifying user agent, runs strictly sequentially with a
1.5 second delay, and makes about 120 requests for the whole catalogue. It **stops on a 401,
403 or 429** rather than retrying under another identity. Petbarn's edge denies some named
crawlers; that is a signal to respect, not an obstacle to route around.

Reviewer nicknames and locations are dropped at ingest and have no field to be stored in.
Raw responses are archived locally for reproducibility and are not committed, because those
untouched bodies still contain them.

## The data

Captured 8 September 2026 from petbarn.com.au and its public Bazaarvoice review API.

| | |
|---|---|
| Products | 9 |
| Ratings | 6,664 |
| Written reviews | 3,054 |
| Written reviews at 3 stars or below | 126 (4.1%) |
| Review bodies flagged as suspicious at ingest | 0 |

## Testing

167 tests, offline, no API key, about 13 seconds. Providers are stubbed; the page is driven
with Streamlit's own headless harness.

The tests aim at the ways this system could state something false rather than at coverage.
Several exist because the bug happened: paging that skipped reviews, the shelf price returned
as the member price, dates a day early in Australian time, an entity-encoded injection that
slipped past the sanitiser, praise reported as a drawback.

Every assertion was checked by breaking the code and confirming the test fails. That habit
found several tests that were passing while the behaviour they named was broken.

## Known limitations

- **Nine products.** The brief asked for eight to ten. Anything else gets a scoped refusal
  naming what is available.
- **Tool A returns Petbarn's description, not a structured specification sheet.** Their
  catalogue API exposes more attributes than the ingest currently requests.
- **Aspect matching is keyword-based**, so it misses paraphrase. Traded for something
  auditable that needs no model and can be corrected by editing a list. Where a topic can be
  denied rather than reported, the denials are counted separately.
- **Per-product aspect vocabularies differ slightly**, because words from a product's own
  name are excluded from its lexicon. Comparisons of raw mention counts between two products
  are therefore approximate.
- **The session question limit is a counter, not a control.** Session state lives in the
  browser and a refresh resets it. On a free tier the exposure is not cost but availability -
  the daily allowance is a few full sessions - and per-IP limiting is not available on this
  host.
- **Review text is untrusted input.** It is sanitised at ingest, the model is told tool
  results are data rather than instructions, and anything that would make a browser fetch
  from another host is stripped before display. The tools are read-only over nine products,
  so a successful injection buys a wrong paragraph about dog food.

## Data provenance

A one-off, low-volume, non-commercial capture of publicly available product and review data,
made for a technical assessment. Nine products. Reviewer identifiers are not stored, review
excerpts are capped in length, and the dataset is not redistributed. Happy to remove it on
request.
