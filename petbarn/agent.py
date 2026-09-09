"""One question in, one answer out, with whatever tool calls that took.

The loop is short on purpose: send the conversation, notice the model wants a
tool, run it, hand back the result, repeat until it stops asking. Everything
around that is guard rails, and they exist because each failure below happened
to somebody and produced a confident wrong answer or a blank red box.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from petbarn import prompts
from petbarn.config import LIMITS
from petbarn.providers import Chain, Completion, ProviderUnavailable, ToolCall
from petbarn.resolve import resolve
from petbarn.snapshot import Snapshot
from petbarn.tools import REGISTRY, SCHEMAS, get_product_details, get_product_reviews_and_sentiment

log = logging.getLogger(__name__)

STUB = "[earlier tool result omitted to save room; it is shown in the trace]"


@dataclass
class ToolRun:
    """One tool call, kept for the transcript and for the trace in the UI."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    seconds: float
    resolved_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.result.get("status") == "ok"


@dataclass
class Turn:
    answer: str
    runs: list[ToolRun] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    provider: str = ""
    model: str = ""
    rounds: int = 0
    seconds: float = 0.0
    degraded: bool = False


def _call_tool(call: ToolCall) -> ToolRun:
    """Run one tool, or explain to the model why it could not be run.

    Every failure returns a result rather than raising. A model that is told
    what went wrong can correct itself on the next round; one that receives an
    exception sees nothing at all.
    """
    started = time.monotonic()

    function = REGISTRY.get(call.name)
    if function is None:
        return ToolRun(call.name, call.arguments, {
            "status": "unknown_tool",
            "note": f"There is no tool called {call.name}.",
            "available": sorted(REGISTRY),
        }, time.monotonic() - started)

    # Guarded here as well as in the providers: this runs above the try below,
    # and one provider builds arguments straight from the response body.
    arguments = dict(call.arguments) if isinstance(call.arguments, dict) else {}
    product = arguments.pop("product", None)
    aspect = arguments.pop("aspect", None)

    if arguments:
        log.info("%s: ignoring unexpected arguments %s", call.name, sorted(arguments))

    try:
        if call.name == "get_product_reviews_and_sentiment":
            result = function(product, aspect=aspect)
        else:
            result = function(product)
    except Exception as exc:
        # The tools are written not to raise; this is the net under that claim.
        log.exception("%s raised", call.name)
        result = {"status": "error",
                  "note": "That lookup failed. Try the other tool or another product."}
        return ToolRun(call.name, call.arguments, result, time.monotonic() - started)

    return ToolRun(
        name=call.name,
        arguments=call.arguments,
        result=result,
        seconds=time.monotonic() - started,
        resolved_id=(result.get("product") or {}).get("id"),
    )


def _echo(call: ToolCall) -> dict:
    """The call as it must be sent back on the next request.

    Anything the provider attached to it goes back untouched. Gemini 3 signs
    each call and rejects the follow-up without that signature, so dropping it
    silently reduces every conversation to one round.
    """
    echoed = {
        "id": call.id,
        "type": "function",
        "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
    }
    if call.extra:
        echoed["extra_content"] = call.extra
    return echoed


def _trim(history: list[dict]) -> list[dict]:
    """Keep the conversation, drop the bulk of older tool results.

    A tool result is one to two thousand tokens and the whole transcript is
    resent on every round. Left alone it is the transcript, not any single
    answer, that exhausts a per-minute allowance, and the failure lands three
    questions into a conversation rather than on the first.

    The cut has to respect message groups. An assistant message carrying tool
    calls and the replies to those calls are one unit, and slicing through the
    middle leaves a tool reply with no call before it, which providers reject.
    A rejection looks like a dead provider to the chain, so a bad cut here used
    to disable every provider for the rest of the session.
    """
    kept = history[-(LIMITS.history_turns * 2):]

    # Start at the first user message that is not a tool reply, discarding any
    # partial group the slice landed in the middle of.
    start = next(
        (i for i, m in enumerate(kept)
         if m.get("role") == "user" and not m.get("tool_call_id")),
        len(kept),
    )
    kept = kept[start:]

    # Older tool results are replaced by a stub; the most recent turn keeps its
    # results in full, because that is what the model is still reasoning about.
    boundary = max(
        (i for i, m in enumerate(kept) if m.get("role") == "user"), default=len(kept)
    )

    trimmed: list[dict] = []
    open_calls: set[str] = set()
    for i, message in enumerate(kept):
        if message.get("role") == "assistant":
            open_calls = {c["id"] for c in message.get("tool_calls") or []}
        elif message.get("role") == "tool":
            # Belt and braces: a reply whose call did not survive is dropped
            # rather than sent on its own.
            if message.get("tool_call_id") not in open_calls:
                continue
            if i < boundary:
                message = {**message, "content": STUB}
        trimmed.append(message)

    return trimmed


def _fallback(question: str, snap: Snapshot) -> Turn:
    """What to show when no model will answer.

    The host hides error detail, so an unhandled failure is a blank red box to
    the visitor. This rung has no model in it: it resolves the question against
    the catalogue itself and renders the underlying data, which is worth more
    than an apology and cannot fail the same way.
    """
    found = resolve(question)
    lines = ["The language model is unavailable, so here is the underlying data."]
    runs: list[ToolRun] = []

    if found.status == "resolved":
        for tool in (get_product_details, get_product_reviews_and_sentiment):
            result = tool(found.catalogue_id)
            runs.append(ToolRun(tool.__name__, {"product": found.catalogue_id},
                                result, 0.0, found.catalogue_id))

        details, reviews = runs[0].result, runs[1].result
        product, sentiment = details["product"], reviews["sentiment"]
        lines += [
            "",
            f"**{product['name']}** - ${product['price_aud']}"
            + (f" (members ${product['member_price_aud']})" if product["member_price_aud"] else ""),
            f"{sentiment['mean_rating']} stars from "
            f"{sentiment['all_reviews_including_ratings_only']} reviews, "
            f"{sentiment['share_positive_4_or_5']:.0%} rated four or five.",
            "",
            "Recent comments:",
        ]
        lines += [f"- [{q['rating']}*] {q['text'][:180]}" for q in reviews["quotes"][:3]]
    else:
        lines += ["", "Products in this catalogue:"]
        lines += [f"- {p.display_name}" for p in snap.product_list]

    return Turn(answer="\n".join(lines), runs=runs, provider="none", degraded=True)


def answer(question: str, history: list[dict], chain: Chain, snap: Snapshot) -> Turn:
    """Answer one question, calling tools as the model asks for them."""
    started = time.monotonic()
    question = (question or "").strip()[: LIMITS.max_question_chars]

    messages = [{"role": "system", "content": prompts.system_prompt(snap)}]
    messages += _trim(history)
    messages.append({"role": "user", "content": question})

    chain.begin_turn()

    turn = Turn(answer="")
    seen: set[tuple[str, str]] = set()
    executed = 0

    for round_number in range(1, LIMITS.max_tool_rounds + 1):
        turn.rounds = round_number
        out_of_time = time.monotonic() - started > LIMITS.turn_deadline_seconds
        # The last round, or the one that runs past the deadline, is for prose.
        # Withholding the tools is the request; ignoring any tool calls that
        # come back anyway is what makes it a guarantee.
        final_round = out_of_time or round_number == LIMITS.max_tool_rounds

        remaining = LIMITS.turn_deadline_seconds - (time.monotonic() - started)

        try:
            completion: Completion = chain.complete(
                messages,
                # On the last round, and when out of time, ask for prose from
                # what has already been gathered rather than another tool call
                # that there is no budget left to answer.
                tools=[] if final_round else SCHEMAS,
                # Every provider the chain tries shares what is left of the
                # turn's budget, so walking a chain of slow providers cannot
                # take several times the deadline it is supposed to enforce.
                budget=max(remaining, 1.0),
            )
        except ProviderUnavailable as exc:
            log.warning("no provider could answer: %s", exc)
            fallback = _fallback(question, snap)
            fallback.seconds = time.monotonic() - started
            fallback.runs = turn.runs + fallback.runs
            return fallback

        turn.provider = completion.provider
        turn.model = completion.model

        if final_round or not completion.wants_tools:
            turn.answer = completion.text.strip()
            if turn.answer:
                messages.append({"role": "assistant", "content": turn.answer})
            break

        messages.append({
            "role": "assistant",
            "content": completion.text or None,
            "tool_calls": [_echo(c) for c in completion.tool_calls],
        })

        for call in completion.tool_calls:
            fingerprint = (call.name, json.dumps(call.arguments, sort_keys=True))
            if executed >= LIMITS.max_tool_calls:
                # Round caps alone do not bound this: a model can emit any
                # number of calls per round, and the aspect argument is free
                # text, so varying it defeats the duplicate check.
                result = {"status": "budget_spent",
                          "note": "Enough data has been gathered. Answer with what you have."}
            elif fingerprint in seen:
                # Asking the same question twice means the answer was not
                # understood, and a third attempt will not help.
                result = {"status": "already_answered",
                          "note": "This exact call was already made. Use the result above."}
            else:
                seen.add(fingerprint)
                executed += 1
                run = _call_tool(call)
                turn.runs.append(run)
                result = run.result

            # One reply per call id, including for calls we declined to run:
            # a missing one leaves the conversation malformed for the next round.
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, separators=(",", ":")),
            })

    if not turn.answer:
        turn.answer = (
            "I gathered the data but ran out of steps before writing it up. "
            "Try asking about one product at a time."
        )

    turn.seconds = time.monotonic() - started
    # History for the caller to hand back next time. The system prompt is
    # rebuilt each turn, so it is dropped; the assistant's own reply is kept,
    # because without it a follow-up arrives in a conversation where the
    # assistant never spoke.
    turn.messages = messages[1:]
    return turn
