# Architecture

Two diagrams: where the data lives, and what happens when somebody asks a question.

## The system

Everything slow, rate-limited or liable to fail happens offline and produces a snapshot.
Everything a visitor touches reads that snapshot.

```mermaid
flowchart TB
    subgraph offline["OFFLINE — run by hand, never during a conversation"]
        direction LR
        PP["Product pages<br/><i>JSON-LD: price, brand, stock</i>"]
        CA["Catalogue API<br/><i>descriptions</i>"]
        RA["Review API<br/><i>full history, paged 100 at a time</i>"]
        ING["<b>ingest</b><br/><i>honest user agent<br/>1.5s between requests<br/>validate, then write</i>"]
        RAW["raw archive<br/><i>local, not committed</i>"]
        PP --> ING
        CA --> ING
        RA --> ING
        ING --> RAW
    end

    SNAP["<b>data/snapshot/ — committed to git</b><br/>catalog.json · reviews.jsonl.gz · manifest.json<br/><i>9 products · 3,054 written reviews · captured 8 Sep 2026</i>"]

    subgraph serving["SERVING — every question, no scraping"]
        direction LR
        UI["Streamlit page<br/><i>chat + tool trace</i>"]
        LOOP["<b>agent loop</b><br/><i>4 rounds max</i>"]
        CHAIN["provider chain<br/><i>1 gemini flash-lite<br/>2 gemini flash<br/>3 groq gpt-oss<br/>4 no model at all</i>"]
        TOOLS["two tools<br/><i>details · reviews</i>"]
        ANA["analysis<br/><i>stats · sampler · caveats</i>"]
        UI --> LOOP
        LOOP -->|wants a tool| CHAIN
        LOOP --> TOOLS
        TOOLS --> ANA
    end

    ING ==> SNAP
    SNAP ==>|reads| ANA
```

The snapshot is the seam. Nothing below it can fail in a way a visitor would see, except the
model, and that has three fallbacks — the last of which is not a model.

## One question, end to end

A comparison takes three model calls and two tool calls, in about five seconds.

```mermaid
sequenceDiagram
    participant P as page
    participant L as agent loop
    participant M as model
    participant T as tools + snapshot

    P->>L: "compare Black Hawk and Prime100"
    L->>M: question + catalogue + 2 tool schemas
    M-->>L: call reviews(black-hawk) and reviews(prime100)
    L->>T: resolve id, then read the snapshot
    T-->>L: stats over all 3,054 · 8 quotes · caveats · coverage
    L->>M: both results, appended to the conversation
    M-->>L: prose, quoting the numbers it was given
    L->>P: answer + the record of both tool calls
    Note over P: image markup stripped and non-Petbarn links<br/>reduced to plain text before anything is drawn
```

**The arrows matter more than the boxes.** Everything numeric travels back from the tools already
computed. If the model calculated anything itself, those arrows would carry raw review text
instead, and every figure in the answer would be an estimate nobody could check. Moving that work
to the left of the model is the whole design.

## Why the pieces are where they are

**The tools do the arithmetic.** A model asked how many people complained about price will produce
a number that sounds right, and a reader cannot tell it from one that is right. Computing it in
Python makes that impossible rather than unlikely. It also means a small, fast model is enough.

**Reviews come from a snapshot.** The review API returns a hundred reviews per call, and 96% of
this catalogue is four and five stars, so a live fetch would return almost nothing critical. Paging
the whole history once, offline, is what lets the tool go looking for the rare complaints.

**The chain ends in no model.** The failure worth defending against is not an outage but a mistyped
key or a rate limit, which look identical to a visitor because the host hides error detail. The
last rung resolves the question against the catalogue itself and renders the underlying data.

**Untrusted text is cleaned twice.** Once at ingest, where review bodies are normalised and
suspicious content is flagged, and once at display, because the model composes its answer from that
text and what it writes is not what the tool returned.
