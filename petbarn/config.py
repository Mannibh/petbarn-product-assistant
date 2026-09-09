"""Settings and secrets, read the same way locally and when deployed.

Streamlit keeps deployment secrets in its own store and local ones in
.streamlit/secrets.toml, while a plain shell run has neither. One accessor
covers all three so nothing in the app has to know which it is running under.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


def secret(name: str, default: str | None = None) -> str | None:
    """Look in Streamlit's store first, then the environment.

    The import and the lookup are both guarded. Streamlit raises rather than
    returning nothing when no secrets file exists, which is the normal case for
    a test run or a CLI script, and that must not be an error.
    """
    try:
        import streamlit as st
    except ImportError:
        pass
    else:
        try:
            value = st.secrets.get(name)
        except FileNotFoundError:
            # Normal: a test run or a CLI script has no secrets file. Anything
            # else, such as a malformed TOML, is a real misconfiguration and
            # should not be silently turned into a missing key.
            value = None
        if value:
            return str(value)

    return os.environ.get(name, default)


@dataclass(frozen=True)
class Limits:
    """Caps that each keep the app inside a free tier and make it feel faster.

    Every value here was chosen because it does both jobs. Thrift and
    responsiveness are the same problem when the provider rations tokens per
    minute rather than charging for them.
    """

    max_question_chars: int = 500
    max_tool_rounds: int = 4
    # Rounds alone do not bound the work: a model may emit any number of calls
    # in one round, and each result is over a thousand tokens.
    max_tool_calls: int = 10
    max_output_tokens: int = 900
    # Turns of history resent to the model. Older tool results are replaced by a
    # short stub, because a growing transcript is what actually exhausts a
    # per-minute allowance during a comparison.
    history_turns: int = 6
    turn_deadline_seconds: float = 45.0
    request_timeout_seconds: float = 25.0
    # Shown as a counter, never enforced as security: a new session is one
    # refresh away and pretending otherwise would be dishonest.
    session_turn_budget: int = 25


LIMITS = Limits()
