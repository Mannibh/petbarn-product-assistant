"""The chat interface.

Streamlit re-runs this file from the top on every interaction, so anything that
must survive lives in session_state and everything drawn on screen is drawn
again from scratch each time.

That has one consequence worth stating, because getting it wrong is the classic
bug in a Streamlit chat app. The tool trace cannot be drawn only as the tools
run: the next rerun would redraw the conversation from history and the trace
would silently vanish. So two records are kept side by side. One is the
API-shaped transcript that goes back to the model; the other holds the tool
runs in full, for the screen. They are written together and replayed together.
"""

from __future__ import annotations

import logging

import streamlit as st

from petbarn import snapshot
from petbarn.agent import Turn, answer
from petbarn.config import LIMITS
from petbarn.providers import build_chain
from petbarn.render import safe_markdown

log = logging.getLogger(__name__)

STARTERS = [
    "What are people saying about the price and quality of the Black Hawk lamb and rice?",
    "Compare the reviews between the NexGard Spectra and the Simparica Trio.",
    "What are the main pros and cons of the Breeders Choice cat litter?",
    "What do reviewers say about the price of the Royal Canin Maxi?",
]

st.set_page_config(page_title="Petbarn product assistant", page_icon="🐾", layout="centered")


@st.cache_resource
def load_snapshot():
    """Read once per process rather than once per interaction."""
    return snapshot.get()


def session_chain():
    """One chain per visitor, so a provider resting for one person does not
    rest for everyone."""
    if "chain" not in st.session_state:
        st.session_state.chain = build_chain()
    return st.session_state.chain


def render_trace(runs, expanded: bool) -> None:
    """The tool calls, drawn the same way live and on replay.

    This is what shows that the assistant genuinely called something rather
    than answering from memory, so it is drawn from the stored record and never
    only as it happens.
    """
    for run in runs:
        status = run.result.get("status", "?")
        # Only a genuine fault is drawn as one. Being asked about a product we
        # do not stock is the resolver working, and colouring it red tells the
        # visitor the app broke on a turn where it behaved correctly.
        broke = status in {"error", "unknown_tool"}
        state = "error" if broke else "complete"
        label = f"{safe_markdown(run.name)} · {status} · {run.seconds * 1000:.0f} ms"
        with st.status(label, state=state, expanded=expanded):
            st.caption("Arguments the model produced")
            st.json(run.arguments, expanded=True)
            if run.resolved_id and run.resolved_id != run.arguments.get("product"):
                st.caption(f"Resolved to **{run.resolved_id}**")
            st.caption(f"Result · {run.result.get('status')}")
            st.json(run.result, expanded=False)


def render_answer(turn: Turn, expanded: bool) -> None:
    if turn.runs:
        render_trace(turn.runs, expanded)
    if turn.degraded:
        st.warning("No language model is available, so this is the underlying data.")
    st.markdown(safe_markdown(turn.answer))
    parts = [p for p in (turn.provider, turn.model) if p]
    if parts:
        st.caption(" · ".join([*parts, f"{turn.seconds:.1f}s", f"{turn.rounds} rounds"]))


def sidebar(snap) -> None:
    with st.sidebar:
        st.subheader("This assistant")
        written = snap.total_reviews
        rated = sum(p.summary.total_reviews for p in snap.product_list)
        # Both numbers, because the per-product rows below show the larger one
        # and a single word covering both reads as an arithmetic error.
        st.markdown(
            f"Answers questions about **{len(snap.products)} Petbarn products** using "
            f"**{rated:,} customer ratings**, {written:,} of which have written "
            f"reviews, by calling two tools."
        )

        age = snap.age_days()
        st.caption(
            f"Data captured {snap.captured_at:%d %b %Y}"
            + (" (today)" if age == 0 else f" ({age} day{'s' if age != 1 else ''} ago)")
        )

        st.divider()
        st.caption("Products in the catalogue")
        for product in snap.product_list:
            st.markdown(
                f"- {product.display_name}  \n"
                f"  <span style='color:#888;font-size:0.85em'>"
                f"${product.price_aud:,.2f} · {product.summary.mean_rating:.2f}★ · "
                f"{product.summary.total_reviews:,} reviews</span>",
                unsafe_allow_html=True,
            )

        st.divider()
        chain = session_chain()
        if chain.providers:
            st.caption("Model providers, in order")
            for provider in chain.providers:
                resting = provider.name not in {p.name for p in chain.available}
                st.markdown(f"- {'~~' if resting else ''}`{provider.model}`{'~~' if resting else ''}")
        else:
            st.error("No API key is configured.")

        used = len(st.session_state.transcript)
        st.caption(f"{used} of {LIMITS.session_turn_budget} questions this session")


def main() -> None:
    try:
        snap = load_snapshot()
    except Exception:
        log.exception("snapshot could not be loaded")
        st.error("The product data is unavailable. Please try again shortly.")
        st.stop()
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("transcript", [])

    st.title("Petbarn product assistant")
    st.caption(
        "Ask about product details or what customers think. Every figure comes from a tool "
        "call over real review data, and each call is shown."
    )

    try:
        conversation(snap)
    finally:
        # Last, and on every path including the early returns. The sidebar
        # reports how many questions have been asked, and rendering it before
        # the current one is stored left it permanently one behind.
        sidebar(snap)


def conversation(snap) -> None:
    # Replay first. Everything on screen is redrawn from these records on every
    # interaction, which is why the trace is stored rather than only rendered.
    for exchange in st.session_state.transcript:
        with st.chat_message("user"):
            st.markdown(exchange["question"])
        with st.chat_message("assistant"):
            render_answer(exchange["turn"], expanded=False)

    # The slot is reserved here so the suggestions appear above the input, but
    # it is filled after the input is read. Rendering them first left them on
    # screen beside the first answer, because whether to show them is only known
    # once the question is in hand.
    suggestions = st.empty()

    typed = st.chat_input("Ask about a product...", max_chars=LIMITS.max_question_chars)

    asked = (typed or "").strip() or None
    if not asked and not st.session_state.transcript:
        with suggestions.container():
            st.caption("Try one of these")
            for i, starter in enumerate(STARTERS):
                if st.button(starter, key=f"starter-{i}", use_container_width=True):
                    asked = starter

    if asked:
        # Clearing the slot rather than never filling it, because a clicked
        # suggestion only becomes known after the buttons have been drawn.
        suggestions.empty()

    if not asked:
        return

    if len(st.session_state.transcript) >= LIMITS.session_turn_budget:
        st.warning("That is the limit for one session. Reload the page to start again.")
        return

    with st.chat_message("user"):
        st.markdown(asked)

    with st.chat_message("assistant"):
        with st.spinner("Looking it up..."):
            try:
                turn = answer(asked, st.session_state.history, session_chain(), snap)
            except Exception:
                # The host hides error detail, so an unhandled exception reaches
                # the visitor as a blank red box. Log the detail where it is
                # useful and show something they can act on. History is left
                # untouched so the next question starts from a clean state.
                log.exception("turn failed for question of %d chars", len(asked))
                st.error(
                    "Something went wrong answering that. Try rephrasing, or ask "
                    "about one product at a time."
                )
                return

        # The first trace is open so that a visitor sees the tools fire without
        # having to discover that the rows expand.
        render_answer(turn, expanded=not st.session_state.transcript)

    st.session_state.history = turn.messages
    st.session_state.transcript.append({"question": asked, "turn": turn})


main()
