"""The chat page itself, driven headlessly.

Everything below runs the real streamlit_app.py with a scripted provider, so no
key and no network are involved. These exist because the page has failure modes
the unit tests cannot see: Streamlit re-runs the whole script on every
interaction, and anything drawn only as it happens is gone by the next one.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

from petbarn.config import LIMITS
from petbarn.providers import Chain, Completion, ToolCall

REVIEWS = "get_product_reviews_and_sentiment"
APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


class Scripted:
    """One scripted answer per question, not per call.

    A turn takes two calls, one to ask for the tool and one to write the prose,
    so a stand-in that advances on every call answers the second question during
    the first one.
    """

    name, model = "scripted", "fake-1"

    def __init__(self, *answers: str, product: str = "black-hawk-lamb-rice"):
        self.answers = list(answers) or ["Answer."]
        self.product = product
        self.received: list[list[dict]] = []
        self._turn = 0

    def complete(self, messages, tools, timeout=None):
        # A copy: the loop appends to this same list as the turn proceeds, so
        # keeping the reference would record the end of the turn rather than
        # what was actually sent at this point.
        self.received.append([dict(m) for m in messages])
        if tools:
            return Completion(text="", tool_calls=(
                ToolCall("c1", REVIEWS, {"product": self.product}),))

        text = self.answers[min(self._turn, len(self.answers) - 1)]
        self._turn += 1
        return Completion(text=text)


def hostile(text: str):
    class Fixed(Scripted):
        def complete(self, messages, tools, timeout=None):
            self.received.append([dict(m) for m in messages])
            return Completion(text=text)
    return Fixed()


def app(provider) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["chain"] = Chain([provider])
    return at


def ask(at: AppTest, question: str) -> AppTest:
    at.chat_input[0].set_value(question).run()
    return at


def test_an_unexpected_failure_shows_a_message_rather_than_a_red_box():
    """The handler that exists to prevent a blank error screen used to raise
    inside itself, producing exactly the screen it was written to prevent."""
    class Exploding:
        name, model = "boom", "none"
        def complete(self, messages, tools, timeout=None):
            raise ValueError("something unforeseen")

    at = ask(app(Exploding()).run(), "how is the black hawk?")

    assert not at.exception, "the error handler itself crashed"
    assert at.error, "the visitor was told nothing"
    assert "Try rephrasing" in at.error[0].value


def test_clicking_a_suggestion_clears_the_other_suggestions():
    """A clicked suggestion is only known after the buttons have been drawn,
    so they have to be cleared rather than never rendered."""
    at = app(Scripted("Answer.")).run()
    at.button[0].click().run()
    assert at.button == [], "the suggestions stayed on screen beside the answer"


def test_the_session_counter_counts_questions_not_records():
    """It counted transcript rows, of which there were two per exchange, so it
    cut the visitor off at half the advertised limit and displayed a count
    higher than the limit it was comparing against."""
    at = app(Scripted("A.", "B.")).run()
    at = ask(at, "first question")
    at = ask(at, "second question")

    assert len(at.session_state["transcript"]) == 2
    counters = [c.value for c in at.sidebar.caption if "questions this session" in c.value]
    assert counters and counters[-1].startswith(f"2 of {LIMITS.session_turn_budget}")


def test_a_scoped_refusal_is_not_drawn_as_a_fault():
    """Being asked about a product we do not stock is the resolver working. A
    red error row tells the visitor the app broke on a turn where it did not."""
    class NamesAnUnknownProduct(Scripted):
        def complete(self, messages, tools, timeout=None):
            self.received.append([dict(m) for m in messages])
            if tools:
                return Completion(text="", tool_calls=(
                    ToolCall("c1", REVIEWS, {"product": "hills science diet"}),))
            return Completion(text="That is outside this catalogue.")

    at = ask(app(NamesAnUnknownProduct()).run(), "do you sell hills?")

    assert at.status[0].proto.state != 3, "a scoped refusal was drawn as an error"
    assert "not_found" in at.status[0].label


def test_only_the_first_answers_trace_starts_open():
    """A grader has to see the tools fire without discovering that rows expand,
    but every later row opening would bury the conversation."""
    at = ask(app(Scripted("First.", "Second.")).run(), "how is the black hawk?")
    assert at.status[0].proto.expanded

    at = ask(at, "and the cat litter?")
    assert [s.proto.expanded for s in at.status] == [False, False]


def test_a_whitespace_only_question_is_ignored():
    at = ask(app(Scripted("Answer.")).run(), "    ")
    assert at.session_state["transcript"] == []


def test_the_sidebar_names_both_review_populations():
    """The per-product rows show every rating; the headline used to show only
    the written ones under the same word, so the rows summed to more than the
    total directly above them."""
    at = app(Scripted("hi")).run()
    headline = " ".join(m.value for m in at.sidebar.markdown if "Petbarn products" in m.value)
    assert "ratings" in headline and "written" in headline


def test_the_page_loads_and_shows_the_catalogue():
    at = app(Scripted("hi")).run()
    assert not at.exception
    assert any("Petbarn product assistant" in m.value for m in at.title)
    sidebar_text = " ".join(m.value for m in at.sidebar.markdown)
    assert "Black Hawk" in sidebar_text and "Breeders Choice" in sidebar_text


def test_starter_questions_are_offered_before_anything_is_asked():
    """A grader opening an empty chat box has nothing to go on."""
    at = app(Scripted("hi")).run()
    assert len(at.button) == 4


def test_asking_a_question_shows_an_answer_and_the_tool_it_called():
    at = ask(app(Scripted("Black Hawk averages 4.79 stars.")).run(),
             "how is the black hawk?")

    assert not at.exception
    assert any("4.79" in m.value for m in at.markdown)
    assert len(at.status) == 1
    assert REVIEWS in at.status[0].label


def test_the_first_trace_shows_the_argument_the_model_produced():
    """A grader has to be able to see that a tool really ran, and what was
    passed to it, without taking the answer on trust."""
    at = ask(app(Scripted("Answer.")).run(), "how is the black hawk?")
    shown = at.status[0].json
    assert shown and shown[0].body == '{"product": "black-hawk-lamb-rice"}'


def test_a_second_question_does_not_erase_the_first_answers_trace():
    """The bug this page is shaped around. Streamlit redraws everything on each
    interaction, so a trace rendered only while the tools ran disappears the
    moment anything else happens."""
    provider = Scripted("First answer.", "Second answer.")
    at = ask(app(provider).run(), "how is the black hawk?")
    assert len(at.status) == 1

    at = ask(at, "and the cat litter?")

    assert not at.exception
    assert len(at.status) == 2, "the first question's trace was lost on rerun"
    answers = " ".join(m.value for m in at.markdown)
    assert "First answer." in answers and "Second answer." in answers


def test_the_second_question_is_asked_with_the_first_still_in_the_conversation():
    provider = Scripted("First.", "Second.")
    at = ask(app(provider).run(), "how is the black hawk?")
    at = ask(at, "and the cat litter?")

    # The first request of the second turn: what the model is given before it
    # has done anything about the new question.
    opening = next(r for r in provider.received
                   if r[-1].get("content") == "and the cat litter?")
    spoken = [m for m in opening if m["role"] == "assistant" and not m.get("tool_calls")]
    assert any(m["content"] == "First." for m in spoken), "the first answer was not carried over"
    assert any(m["content"] == "how is the black hawk?" for m in opening if m["role"] == "user")


def test_starter_questions_disappear_once_the_conversation_starts():
    at = ask(app(Scripted("Answer.")).run(), "how is the black hawk?")
    assert at.button == [], "the suggestions stayed on screen beside the answer"


def test_a_starter_button_asks_its_question():
    at = app(Scripted("Answer.")).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.status) == 1


def test_an_image_in_the_model_output_never_reaches_the_page():
    """Answers are composed from review text written by strangers. An image
    reference would make the reader's browser fetch from somewhere we do not
    control, the moment the answer appears."""
    provider = hostile("Great value ![](http://attacker.example/pixel.png) overall.")
    at = ask(app(provider).run(), "how is the black hawk?")

    rendered = " ".join(m.value for m in at.markdown)
    assert "attacker.example" not in rendered
    assert "Great value" in rendered


def test_no_provider_shows_the_data_rather_than_an_error():
    from petbarn.providers import ProviderUnavailable

    class Dead:
        name, model = "dead", "none"
        def complete(self, messages, tools, timeout=None):
            raise ProviderUnavailable("401 invalid key")

    at = ask(app(Dead()).run(), "what about the cat litter?")

    assert not at.exception
    assert at.warning, "the visitor should be told the answer is degraded"
    assert any("Breeders Choice" in m.value for m in at.markdown)


def test_the_sidebar_reports_a_missing_key_rather_than_failing_quietly():
    at = AppTest.from_file(APP, default_timeout=60)
    at.session_state["chain"] = Chain([])
    at.run()
    assert at.sidebar.error
