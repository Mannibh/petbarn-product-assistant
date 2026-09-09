"""The tool loop, driven by a scripted provider so no key or network is needed.

Every test here is a failure that produces either a confident wrong answer or a
blank error box, which are the two outcomes this project cannot afford.
"""

from __future__ import annotations

import json

import pytest

from petbarn import snapshot
from petbarn.agent import STUB, _trim, answer
from petbarn.config import LIMITS
from petbarn.providers import Chain, Completion, ProviderUnavailable, ToolCall

REVIEWS = "get_product_reviews_and_sentiment"


class Scripted:
    """A stand-in model that returns whatever the script says, in order."""

    name, model = "scripted", "fake-1"

    def __init__(self, *script):
        self.script = list(script)
        self.seen_tools: list[int] = []

    def complete(self, messages, tools, timeout=None):
        self.seen_tools.append(len(tools))
        step = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        return step(tools) if callable(step) else step


def calls(name=REVIEWS, **arguments):
    return Completion(text="", tool_calls=(ToolCall("c1", name, arguments),))


@pytest.fixture(scope="module")
def snap():
    return snapshot.get()


def test_a_tool_call_then_an_answer(snap):
    provider = Scripted(calls(product="black-hawk-lamb-rice"),
                        Completion(text="They rate it 4.79."))
    turn = answer("how is it?", [], Chain([provider]), snap)
    assert turn.answer == "They rate it 4.79."
    assert [r.resolved_id for r in turn.runs] == ["black-hawk-lamb-rice"]


def test_a_comparison_runs_the_tool_once_per_product(snap):
    provider = Scripted(
        Completion(text="", tool_calls=(
            ToolCall("a", REVIEWS, {"product": "black-hawk-lamb-rice"}),
            ToolCall("b", REVIEWS, {"product": "prime100-roo-roll"}))),
        Completion(text="Both average 4.79."))
    turn = answer("compare them", [], Chain([provider]), snap)
    assert sorted(r.resolved_id for r in turn.runs) == ["black-hawk-lamb-rice", "prime100-roo-roll"]


def test_an_invented_tool_is_reported_back_rather_than_crashing(snap):
    provider = Scripted(calls("get_price_history", product="black-hawk-lamb-rice"),
                        Completion(text="No price history."))
    turn = answer("price history?", [], Chain([provider]), snap)
    assert turn.runs[0].result["status"] == "unknown_tool"
    assert "get_product_details" in turn.runs[0].result["available"]


def test_an_identical_repeated_call_is_refused(snap):
    """A model asking the same thing twice did not understand the first answer,
    and a third attempt costs tokens without changing anything."""
    provider = Scripted(calls(product="black-hawk-lamb-rice"),
                        calls(product="black-hawk-lamb-rice"),
                        Completion(text="Done."))
    turn = answer("again", [], Chain([provider]), snap)
    assert len(turn.runs) == 1


def test_every_tool_call_gets_exactly_one_reply(snap):
    """A missing reply leaves the conversation malformed and the next round
    fails for a reason that has nothing to do with the question."""
    provider = Scripted(
        Completion(text="", tool_calls=(
            ToolCall("a", REVIEWS, {"product": "black-hawk-lamb-rice"}),
            ToolCall("b", "no_such_tool", {}))),
        Completion(text="Done."))
    turn = answer("two calls", [], Chain([provider]), snap)
    replies = [m for m in turn.messages if m["role"] == "tool"]
    assert sorted(m["tool_call_id"] for m in replies) == ["a", "b"]


def test_the_loop_stops_and_still_answers(snap):
    """A model that keeps asking for tools must not spin. The last round
    withholds them, and any tool call that comes back anyway is ignored."""
    def greedy(tools):
        if tools:
            return calls(product="black-hawk-lamb-rice")
        return Completion(text="Black Hawk averages 4.79.")

    provider = Scripted(greedy)
    turn = answer("go on", [], Chain([provider]), snap)
    assert turn.rounds == LIMITS.max_tool_rounds
    assert turn.answer == "Black Hawk averages 4.79."
    assert provider.seen_tools[-1] == 0


def test_a_model_that_ignores_the_withheld_tools_is_still_stopped(snap):
    """Withholding the tools on the last round is a request, not a guarantee.
    A model that asks anyway must not send the loop round again, and whatever
    prose it did produce is the answer."""
    stubborn = Scripted(Completion(
        text="Black Hawk averages 4.79 stars.",
        tool_calls=(ToolCall("c1", REVIEWS, {"product": "black-hawk-lamb-rice"}),)))

    turn = answer("go on", [], Chain([stubborn]), snap)

    assert turn.rounds == LIMITS.max_tool_rounds
    assert turn.answer == "Black Hawk averages 4.79 stars."
    assert stubborn.seen_tools[-1] == 0


def test_no_provider_falls_back_to_the_data_itself(snap):
    """The host hides error detail, so an unhandled failure is a blank red box.
    This rung has no model in it and cannot fail the same way."""
    class Dead:
        name, model = "dead", "none"
        def complete(self, messages, tools, timeout=None):
            raise ProviderUnavailable("401 invalid key")

    turn = answer("what about the cat litter?", [], Chain([Dead()]), snap)
    assert turn.degraded
    assert "Breeders Choice Cat Litter" in turn.answer
    assert "4.73" in turn.answer


def test_the_fallback_lists_the_catalogue_when_no_product_was_named(snap):
    class Dead:
        name, model = "dead", "none"
        def complete(self, messages, tools, timeout=None):
            raise ProviderUnavailable("down")

    turn = answer("hello there", [], Chain([Dead()]), snap)
    assert turn.degraded
    assert "Royal Canin" in turn.answer


def test_an_overlong_question_is_cut_before_it_is_sent(snap):
    """Asserting on what the provider received, not on the constant: the
    original version of this test would have passed with the cut removed."""
    seen = {}

    class Recording:
        name, model = "recording", "fake"
        def complete(self, messages, tools, timeout=None):
            seen["user"] = messages[-1]["content"]
            return Completion(text="ok")

    answer("x" * 5000, [], Chain([Recording()]), snap)
    assert len(seen["user"]) == LIMITS.max_question_chars


def test_the_aspect_argument_reaches_the_tool(snap):
    """One line routes it, and deleting that line left the whole suite green."""
    provider = Scripted(
        calls(product="black-hawk-lamb-rice", aspect="puppy"),
        Completion(text="done"))
    turn = answer("anything about puppies?", [], Chain([provider]), snap)
    assert turn.runs[0].result["sentiment"]["requested_topic"]["topic"] == "puppy"


def test_running_out_of_steps_still_says_something(snap):
    """The last resort when the model produced no prose at all."""
    provider = Scripted(Completion(text="", tool_calls=()))
    turn = answer("hello", [], Chain([provider]), snap)
    assert "ran out of steps" in turn.answer


def test_the_fallback_survives_a_tool_that_has_no_reviews(snap):
    """The rung whose docstring says it cannot fail the same way should be held
    to that."""
    class Dead:
        name, model = "dead", "none"
        def complete(self, messages, tools, timeout=None):
            raise ProviderUnavailable("down")

    for question in ("", "!!!", "black hawk", "compare black hawk and prime100"):
        turn = answer(question, [], Chain([Dead()]), snap)
        assert turn.degraded and turn.answer


def _turn(n: int, calls: int = 1, size: int = 4000) -> list[dict]:
    """One well-formed exchange: a question, an assistant asking for tools, a
    reply per call, then the assistant's answer."""
    ids = [f"t{n}-{i}" for i in range(calls)]
    return [
        {"role": "user", "content": f"question {n}"},
        {"role": "assistant", "content": None,
         "tool_calls": [{"id": i, "type": "function",
                         "function": {"name": REVIEWS, "arguments": "{}"}} for i in ids]},
        *({"role": "tool", "tool_call_id": i, "content": "x" * size} for i in ids),
        {"role": "assistant", "content": f"answer {n}"},
    ]


def _malformed(messages: list[dict]) -> list[str]:
    """Tool replies with no call before them, which providers reject."""
    open_ids: set[str] = set()
    problems = []
    for message in messages:
        if message.get("role") == "assistant":
            open_ids = {c["id"] for c in message.get("tool_calls") or []}
        elif message.get("role") == "tool" and message["tool_call_id"] not in open_ids:
            problems.append(message["tool_call_id"])
    return problems


def test_older_tool_results_are_stubbed_out_of_the_transcript():
    """A tool result is over a thousand tokens and the whole transcript is
    resent every round, so it is the history that exhausts a per-minute
    allowance, three questions in rather than on the first."""
    history = _turn(1) + _turn(2)
    trimmed = _trim(history)

    stubbed = [m for m in trimmed if m.get("role") == "tool" and m["content"] == STUB]
    full = [m for m in trimmed if m.get("role") == "tool" and m["content"] != STUB]
    assert len(stubbed) == 1 and len(full) == 1
    assert full[0]["tool_call_id"] == "t2-0"


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 1, 2), (1, 2, 2, 1), (2, 2, 2, 2, 2)])
def test_trimming_never_produces_an_orphaned_tool_reply(shape):
    """An assistant message and the replies to its calls are one unit. Cutting
    through the middle leaves a reply with no call, which providers reject as a
    malformed request. That looked like a dead provider to the chain, so a bad
    cut here used to disable every provider for the rest of the session."""
    history = [m for n, calls in enumerate(shape) for m in _turn(n, calls)]
    assert _malformed(_trim(history)) == []


def test_trimming_starts_at_a_whole_exchange():
    """The slice must not begin inside a group. Verified on its own, because the
    orphan filter below would otherwise hide a broken cut."""
    history = [m for n in range(5) for m in _turn(n, calls=2)]
    trimmed = _trim(history)

    assert trimmed[0]["role"] == "user"
    assert len(trimmed) < len(history)


def test_a_tool_reply_with_no_call_is_dropped():
    """The second defence, tested on its own: history arriving already malformed
    from anywhere must not be passed on to a provider."""
    history = [
        {"role": "user", "content": "q"},
        {"role": "tool", "tool_call_id": "orphan", "content": "result"},
        {"role": "assistant", "content": "a"},
    ]
    assert _malformed(_trim(history)) == []
    assert not any(m["role"] == "tool" for m in _trim(history))


def test_the_assistant_answer_is_kept_in_the_transcript(snap):
    """A follow-up question must not arrive in a conversation where the
    assistant never spoke."""
    provider = Scripted(calls(product="black-hawk-lamb-rice"),
                        Completion(text="They rate it 4.79."))
    turn = answer("how is it?", [], Chain([provider]), snap)

    spoken = [m for m in turn.messages
              if m["role"] == "assistant" and not m.get("tool_calls")]
    assert [m["content"] for m in spoken] == ["They rate it 4.79."]
    assert _malformed(turn.messages) == []


def test_a_two_turn_conversation_stays_well_formed(snap):
    """The UI feeds turn.messages back as history, so the second request has to
    be valid after trimming."""
    first = answer("how is the black hawk?", [],
                   Chain([Scripted(calls(product="black-hawk-lamb-rice"),
                                   Completion(text="4.79 stars."))]), snap)
    second = answer("and the litter?", first.messages,
                    Chain([Scripted(calls(product="breeders-choice-litter"),
                                    Completion(text="4.73 stars."))]), snap)
    assert _malformed(second.messages) == []


def test_tool_calls_are_capped_within_a_turn(snap):
    """Round caps do not bound the work: a model can emit any number of calls
    per round, and varying the free-text aspect defeats the duplicate check."""
    rounds = iter(range(100))

    def greedy(tools):
        if not tools:
            return Completion(text="done")
        # A fresh aspect every round, so the duplicate guard never fires and the
        # only thing standing between this and an unbounded transcript is the
        # call budget.
        n = next(rounds)
        return Completion(text="", tool_calls=tuple(
            ToolCall(f"c{n}-{i}", REVIEWS,
                     {"product": "black-hawk-lamb-rice", "aspect": f"round {n} topic {i}"})
            for i in range(8)))

    turn = answer("everything", [], Chain([Scripted(greedy)]), snap)
    assert len(turn.runs) == LIMITS.max_tool_calls


@pytest.mark.parametrize("junk", [["a"], "text", None, 7])
def test_non_dict_tool_arguments_do_not_escape_the_loop(snap, junk):
    """One provider builds arguments straight from the response body, so this
    is reachable, and it used to raise past the model-free fallback."""
    provider = Scripted(
        Completion(text="", tool_calls=(ToolCall("c1", REVIEWS, junk),)),
        Completion(text="I could not read that request."))
    turn = answer("odd", [], Chain([provider]), snap)
    assert turn.answer
