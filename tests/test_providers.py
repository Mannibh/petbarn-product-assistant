"""The provider layer, exercised without a network or a key.

The Anthropic translation matters most here. It is the only place the two wire
formats genuinely differ, and it is what makes this an abstraction rather than
a base URL that changes.
"""

from __future__ import annotations

import pytest

from petbarn.providers import (
    Anthropic,
    Chain,
    Completion,
    OpenAICompatible,
    ProviderUnavailable,
    ToolCall,
)


class Stub:
    def __init__(self, name, result=None, error=None):
        self.name, self.model, self.calls = name, f"{name}-1", 0
        self._result, self._error = result, error

    def complete(self, messages, tools, timeout=None):
        self.calls += 1
        self.timeout = timeout
        if self._error:
            raise ProviderUnavailable(self._error)
        return self._result or Completion(text="ok", model=self.model)


def test_the_first_working_provider_answers():
    first, second = Stub("first"), Stub("second")
    assert Chain([first, second]).complete([], []).text == "ok"
    assert second.calls == 0


def test_a_failed_provider_rests_rather_than_being_tried_again_immediately():
    """A dead provider should cost one wait, not one per question."""
    dead, alive = Stub("dead", error="401 invalid key"), Stub("alive")
    chain = Chain([dead, alive])

    chain.complete([], [])
    chain.complete([], [])

    assert dead.calls == 1
    assert alive.calls == 2
    assert [p.name for p in chain.available] == ["alive"]


def test_a_rested_provider_is_tried_again_after_a_few_turns():
    """Resting it forever means one rate limit, or one malformed request,
    disables the app for the rest of the session."""
    dead, alive = Stub("dead", error="429 rate limited"), Stub("alive")
    chain = Chain([dead, alive], cooldown_turns=2)

    chain.complete([], [])
    assert [p.name for p in chain.available] == ["alive"]

    chain.begin_turn()
    chain.begin_turn()
    assert [p.name for p in chain.available] == ["dead", "alive"]


def test_the_chain_shares_one_time_budget_across_providers():
    """The turn deadline is only honoured if walking a chain of slow providers
    cannot take several times it."""
    first, second = Stub("first", error="timeout"), Stub("second")
    Chain([first, second]).complete([], [], budget=10.0)

    assert first.timeout == pytest.approx(10.0, abs=0.5)
    assert second.timeout is not None and second.timeout < 10.0


def test_an_authentication_failure_moves_on_rather_than_stopping():
    """Normally a 401 is not worth retrying. Here the likeliest cause is a
    mistyped key in a deployment dashboard, and the visitor must still get an
    answer rather than a blank error box."""
    chain = Chain([Stub("bad-key", error="401"), Stub("good")])
    assert chain.complete([], []).text == "ok"


def test_every_provider_down_raises_with_all_the_reasons():
    chain = Chain([Stub("a", error="429 rate limited"), Stub("b", error="500")])
    with pytest.raises(ProviderUnavailable) as caught:
        chain.complete([], [])
    assert "429" in str(caught.value) and "500" in str(caught.value)


def test_an_empty_chain_fails_immediately():
    with pytest.raises(ProviderUnavailable, match="no provider"):
        Chain([]).complete([], [])


def test_anthropic_lifts_the_system_prompt_out_of_the_messages():
    """Anthropic takes the system prompt as its own argument; leaving it in the
    message list would send it as a user turn."""
    system, turns = Anthropic._translate([
        {"role": "system", "content": "be helpful"},
        {"role": "user", "content": "hello"},
    ])
    assert system == "be helpful"
    assert turns == [{"role": "user", "content": "hello"}]


def test_anthropic_turns_a_tool_result_into_a_user_content_block():
    """There is no tool role in Anthropic's format: a result is a block inside a
    user turn, keyed to the call it answers."""
    _, turns = Anthropic._translate([
        {"role": "tool", "tool_call_id": "call_1", "content": '{"status":"ok"}'},
    ])
    block = turns[0]["content"][0]
    assert turns[0]["role"] == "user"
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "call_1"


def test_anthropic_converts_tool_calls_and_parses_their_arguments():
    _, turns = Anthropic._translate([{
        "role": "assistant",
        "content": "checking",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_product_details", "arguments": '{"product": "royal-canin-maxi"}'}}],
    }])
    text, use = turns[0]["content"]
    assert text == {"type": "text", "text": "checking"}
    assert use["type"] == "tool_use" and use["name"] == "get_product_details"
    assert use["input"] == {"product": "royal-canin-maxi"}


def test_anthropic_tool_input_that_is_not_an_object_becomes_an_empty_dict():
    """Anthropic returns the model's tool input as parsed JSON, which is not
    guaranteed to be an object. The OpenAI path coerces this; without the same
    guard here the two providers behave differently and the loop raises."""
    import httpx

    from petbarn import providers

    class Response:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"content": [{"type": "tool_use", "id": "t1",
                                 "name": "get_product_details",
                                 "input": ["royal-canin-maxi"]}],
                    "usage": {"input_tokens": 1, "output_tokens": 1}}

    original = httpx.post
    httpx.post = lambda *a, **k: Response()
    try:
        completion = Anthropic("anthropic", "claude-haiku-4-5", "key").complete([], [])
    finally:
        httpx.post = original

    assert completion.tool_calls[0].arguments == {}


def test_malformed_tool_arguments_become_an_empty_dict():
    """A truncated response leaves invalid JSON in the arguments field. The loop
    can tell the model that; a parse error mid-translation tells nobody."""
    _, turns = Anthropic._translate([{
        "role": "assistant", "content": "",
        "tool_calls": [{"id": "c1", "type": "function", "function": {
            "name": "get_product_details", "arguments": '{"product": "roy'}}],
    }])
    assert turns[0]["content"][0]["input"] == {}


class FakeOpenAI:
    """Enough of the OpenAI client to drive OpenACompatible.complete."""

    captured: dict = {}

    def __init__(self, **kwargs):
        FakeOpenAI.captured["client"] = kwargs
        self.chat = self

    @property
    def completions(self):
        return self

    def create(self, **kwargs):
        FakeOpenAI.captured["request"] = kwargs
        return _Response()


class _Response:
    class _Function:
        name = "get_product_details"
        arguments = '{"product": "royal-canin-maxi"}'

    class _Call:
        id = "call_1"
        function = _Response._Function() if False else None

    def __init__(self):
        # model_dump is how the real SDK object exposes provider extras such as
        # Gemini's thought signature, so the stand-in has to offer it too.
        call = type("Call", (), {
            "id": "call_1",
            "function": _Response._Function(),
            "model_dump": lambda self: {
                "id": "call_1",
                "extra_content": {"google": {"thought_signature": "sig-abc"}},
            },
        })()
        message = type("Msg", (), {"content": "here you are", "tool_calls": [call]})()
        self.choices = [type("Choice", (), {"message": message})()]
        self.usage = type("Usage", (), {"prompt_tokens": 11, "completion_tokens": 22})()


def test_the_openai_compatible_client_sends_and_parses_a_real_shaped_call(monkeypatch):
    """Neither provider's complete() had a test: replacing both bodies with a
    raised exception left the whole suite green."""
    import openai

    monkeypatch.setattr(openai, "OpenAI", FakeOpenAI)
    provider = OpenAICompatible("gemini", "gemini-2.5-flash", "https://example/v1", "k")

    completion = provider.complete([{"role": "user", "content": "hi"}], [{"t": 1}], timeout=9)

    assert FakeOpenAI.captured["client"]["base_url"] == "https://example/v1"
    assert FakeOpenAI.captured["client"]["timeout"] == 9
    assert FakeOpenAI.captured["client"]["max_retries"] == 0
    assert FakeOpenAI.captured["request"]["model"] == "gemini-2.5-flash"
    assert completion.text == "here you are"
    assert completion.provider == "gemini"
    assert completion.tool_calls[0].arguments == {"product": "royal-canin-maxi"}
    assert (completion.input_tokens, completion.output_tokens) == (11, 22)
    # Gemini 3 rejects the follow-up request without this, which silently turns
    # a multi-round conversation into a single round.
    assert completion.tool_calls[0].extra == {"google": {"thought_signature": "sig-abc"}}


def test_an_sdk_failure_becomes_provider_unavailable(monkeypatch):
    """Anything else stops the chain instead of moving it on."""
    import openai

    def explode(**kwargs):
        raise RuntimeError("connection reset")

    monkeypatch.setattr(openai, "OpenAI", explode)
    with pytest.raises(ProviderUnavailable, match="connection reset"):
        OpenAICompatible("groq", "m", "u", "k").complete([], [])


def test_anthropic_sends_the_documented_request_and_parses_the_reply(monkeypatch):
    import httpx

    sent = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return {"content": [{"type": "text", "text": "hello"},
                                {"type": "tool_use", "id": "t1", "name": "get_product_details",
                                 "input": {"product": "royal-canin-maxi"}}],
                    "usage": {"input_tokens": 5, "output_tokens": 6}}

    def fake_post(url, **kwargs):
        sent.update(url=url, **kwargs)
        return Reply()

    monkeypatch.setattr(httpx, "post", fake_post)
    tools = [{"function": {"name": "get_product_details", "description": "d",
                           "parameters": {"type": "object"}}}]

    completion = Anthropic("anthropic", "claude-haiku-4-5", "secret").complete(
        [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}], tools)

    assert sent["headers"]["x-api-key"] == "secret"
    assert sent["json"]["system"] == "rules"
    assert sent["json"]["tools"][0]["input_schema"] == {"type": "object"}
    assert "max_tokens" in sent["json"]
    assert completion.text == "hello"
    assert completion.provider == "anthropic"
    assert completion.tool_calls[0].arguments == {"product": "royal-canin-maxi"}


def test_parallel_tool_results_go_into_one_user_turn():
    """Anthropic documents the results of one round as blocks in a single turn,
    not as consecutive user messages."""
    _, turns = Anthropic._translate([
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function", "function": {"name": "t", "arguments": "{}"}},
            {"id": "b", "type": "function", "function": {"name": "t", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "a", "content": "one"},
        {"role": "tool", "tool_call_id": "b", "content": "two"},
    ])
    assert [t["role"] for t in turns] == ["assistant", "user"]
    assert [b["tool_use_id"] for b in turns[-1]["content"]] == ["a", "b"]
