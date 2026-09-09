"""Talking to a language model, with somewhere to fall back to.

Two implementations, deliberately. Gemini, Groq, OpenRouter and OpenAI all
accept the OpenAI request format, so one class with a different base URL covers
all four. Anthropic uses a different shape entirely, and supporting it properly
rather than through their compatibility endpoint is what makes this an
abstraction rather than a configuration value. It also keeps the door open: the
app runs on a free tier today and takes a Claude key later without a rewrite.

The failure worth defending against is not a model outage, which is rare. It is
a mistyped key in a deployment dashboard, a rate limit, or a slow response, and
to a visitor those look identical because the host hides error detail. So the
chain moves on from almost anything, including an authentication error, which
is normally a reason to stop.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

from petbarn.config import LIMITS, secret

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    # Provider-specific data attached to the call that has to be handed back
    # verbatim on the next request. Gemini 3 puts a thought signature here and
    # rejects the follow-up with a 400 if it is missing, which turns every
    # multi-round conversation into a single-round one.
    extra: dict[str, Any] | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    tool_calls: tuple[ToolCall, ...] = ()
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ProviderUnavailable(RuntimeError):
    """This provider cannot serve the request. Try the next one."""


class Provider(Protocol):
    name: str
    model: str

    def complete(self, messages: list[dict], tools: list[dict],
                 timeout: float | None = None) -> Completion: ...


def _parse_arguments(raw: str | None, where: str) -> dict[str, Any]:
    """Tool arguments arrive as a JSON string and are not guaranteed to be valid.

    A truncated or malformed call is reported to the loop as an empty argument
    set rather than raising, so the loop can tell the model what went wrong and
    let it try again.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("%s produced unparseable tool arguments: %r", where, raw[:120])
        return {}
    return parsed if isinstance(parsed, dict) else {}


class OpenAICompatible:
    """Gemini, Groq, OpenRouter and OpenAI, which all accept this request shape."""

    def __init__(self, name: str, model: str, base_url: str, api_key: str) -> None:
        self.name = name
        self.model = model
        self._base_url = base_url
        self._api_key = api_key

    def complete(self, messages: list[dict], tools: list[dict],
                 timeout: float | None = None) -> Completion:
        from openai import OpenAI

        try:
            # Construction is inside the try as well: a malformed base URL or a
            # key the SDK rejects on sight fails here, and outside the try that
            # escapes as a raw exception and stops the chain instead of moving
            # it on to the next provider.
            client = OpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=timeout or LIMITS.request_timeout_seconds,
                # One attempt. The SDK default retries twice against a ten
                # minute timeout, which in a chat window is a hang.
                max_retries=0,
            )
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                temperature=0.2,
                max_tokens=LIMITS.max_output_tokens,
            )
        except Exception as exc:
            raise ProviderUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc

        choice = response.choices[0].message
        calls = tuple(
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=_parse_arguments(call.function.arguments, self.name),
                extra=call.model_dump().get("extra_content"),
            )
            for call in (choice.tool_calls or [])
        )
        usage = response.usage
        return Completion(
            text=choice.content or "",
            tool_calls=calls,
            provider=self.name,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )


class Anthropic:
    """Claude's own Messages API.

    Not their OpenAI-compatible endpoint, which their documentation describes as
    a way to evaluate models rather than a production path, and which ignores
    the strict flag on tool schemas.

    The translation below is the reason this class exists. Anthropic takes the
    system prompt as a separate argument rather than a message, and represents
    tool calls and their results as content blocks inside user and assistant
    turns rather than as a separate role.
    """

    API = "https://api.anthropic.com/v1/messages"
    VERSION = "2023-06-01"

    def __init__(self, name: str, model: str, api_key: str) -> None:
        self.name = name
        self.model = model
        self._api_key = api_key

    @staticmethod
    def _translate(messages: list[dict]) -> tuple[str, list[dict]]:
        system: list[str] = []
        turns: list[dict] = []

        for message in messages:
            role = message.get("role")

            if role == "system":
                system.append(message.get("content") or "")

            elif role == "tool":
                block = {
                    "type": "tool_result",
                    "tool_use_id": message["tool_call_id"],
                    "content": message.get("content") or "",
                }
                # A tool result is a block inside a user turn, not its own role,
                # and the results of one round belong in a single turn. Two
                # parallel calls emitted as two consecutive user messages is not
                # the shape Anthropic documents for this.
                if turns and turns[-1]["role"] == "user" and isinstance(turns[-1]["content"], list):
                    turns[-1]["content"].append(block)
                else:
                    turns.append({"role": "user", "content": [block]})

            elif role == "assistant" and message.get("tool_calls"):
                blocks: list[dict] = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    blocks.append({
                        "type": "tool_use",
                        "id": call["id"],
                        "name": call["function"]["name"],
                        "input": _parse_arguments(call["function"]["arguments"], "anthropic"),
                    })
                turns.append({"role": "assistant", "content": blocks})

            else:
                turns.append({"role": role, "content": message.get("content") or ""})

        return "\n\n".join(s for s in system if s), turns

    def complete(self, messages: list[dict], tools: list[dict],
                 timeout: float | None = None) -> Completion:
        system, turns = self._translate(messages)

        payload = {
            "model": self.model,
            "system": system,
            "messages": turns,
            "max_tokens": LIMITS.max_output_tokens,
            "temperature": 0.2,
            "tools": [
                {
                    "name": t["function"]["name"],
                    "description": t["function"]["description"],
                    "input_schema": t["function"]["parameters"],
                }
                for t in tools
            ],
        }

        try:
            response = httpx.post(
                self.API,
                json=payload,
                headers={
                    "x-api-key": self._api_key,
                    "anthropic-version": self.VERSION,
                    "content-type": "application/json",
                },
                timeout=timeout or LIMITS.request_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:
            raise ProviderUnavailable(f"{self.name}: {type(exc).__name__}: {exc}") from exc

        text = "".join(b.get("text", "") for b in body.get("content", [])
                       if b.get("type") == "text")
        calls = tuple(
            ToolCall(id=b["id"], name=b["name"],
                     arguments=b["input"] if isinstance(b.get("input"), dict) else {})
            for b in body.get("content", []) if b.get("type") == "tool_use"
        )
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            tool_calls=calls,
            provider=self.name,
            model=self.model,
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
        )


@dataclass
class Chain:
    """Providers in order of preference, with the failed ones remembered.

    Once a rung fails it is skipped for the rest of the session, so the wait for
    a dead provider is paid once rather than on every question.
    """

    providers: list[Provider] = field(default_factory=list)
    cooldown_turns: int = 3
    _resting: dict[str, int] = field(default_factory=dict)

    @property
    def available(self) -> list[Provider]:
        return [p for p in self.providers if p.name not in self._resting]

    def begin_turn(self) -> None:
        """Age the rest list. A provider that failed is skipped for a few turns
        rather than for the session: a rate limit clears, and one malformed
        request should not be able to blind the app permanently."""
        self._resting = {
            name: turns - 1 for name, turns in self._resting.items() if turns > 1
        }

    def complete(self, messages: list[dict], tools: list[dict],
                 budget: float | None = None) -> Completion:
        """Try each provider in turn, sharing one time budget between them."""
        if not self.providers:
            raise ProviderUnavailable("no provider is configured")

        started = time.monotonic()
        problems: list[str] = []

        for provider in self.available:
            left = None if budget is None else budget - (time.monotonic() - started)
            if left is not None and left <= 0:
                problems.append("ran out of time before trying " + provider.name)
                break
            try:
                return provider.complete(messages, tools, timeout=left)
            except ProviderUnavailable as exc:
                log.warning("%s unavailable, resting it: %s", provider.name, exc)
                problems.append(str(exc))
                self._resting[provider.name] = self.cooldown_turns

        raise ProviderUnavailable("; ".join(problems) or "every provider has failed")


# Ordered by how the app feels to someone waiting, then by headroom.
#
# The lite model is first because it is five times faster on a comparison, and
# measurably so: three runs each gave 4.9-5.5 seconds against 10.7-28.5 for
# gemini-3.6-flash and 13.8-23.6 for gemini-3.5-flash. Its answers are no worse
# here because the tools do the arithmetic and the model only has to read the
# numbers and write them up, which is what a small model is good at.
#
# Groq is second rather than first despite being quick: its free tier rations
# 8,000 tokens a minute, and a comparison resends a growing transcript several
# times, so it fails partway through an answer already on screen. That reads as
# the tools breaking, which is worse than waiting.
#
# gemini-2.5-flash is deliberately absent. It is still listed by the models
# endpoint but returns 404 to keys created recently, which is a good reminder
# that a model list is not a promise.
CANDIDATES = (
    ("gemini", "GEMINI_API_KEY", "gemini-3.5-flash-lite",
     "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("gemini-flash", "GEMINI_API_KEY", "gemini-3.6-flash",
     "https://generativelanguage.googleapis.com/v1beta/openai/"),
    ("groq", "GROQ_API_KEY", "openai/gpt-oss-120b",
     "https://api.groq.com/openai/v1"),
    ("openrouter", "OPENROUTER_API_KEY", "google/gemini-3.5-flash",
     "https://openrouter.ai/api/v1"),
)


def build_chain() -> Chain:
    """Every provider we have a key for, best headroom first."""
    providers: list[Provider] = []

    for name, env_var, model, base_url in CANDIDATES:
        key = secret(env_var)
        if key:
            providers.append(OpenAICompatible(name, model, base_url, key))

    anthropic_key = secret("ANTHROPIC_API_KEY")
    if anthropic_key:
        providers.append(Anthropic("anthropic", "claude-haiku-4-5", anthropic_key))

    return Chain(providers)
