"""Polite HTTP client used by the ingest script.

Three jobs: identify us honestly, go slowly, and keep a copy of everything that
comes back.

The archive is not a nicety. Every response is written to disk exactly as it
arrived, so a parser bug found next week can be fixed and re-run against the
saved bytes instead of hitting Petbarn again.
"""

from __future__ import annotations

import gzip
import logging
import os
import time
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

# Identifies the project and points at a public page describing it, so anyone
# reading their server logs can find out what we are and raise an issue. A
# repository URL rather than a personal address: it is equally contactable and
# puts no personal data into someone else's logs or into this public repo.
PROJECT_URL = "https://github.com/Mannibh/petbarn-product-assistant"
USER_AGENT = os.environ.get(
    "PETBARN_USER_AGENT",
    f"petbarn-product-assistant/0.1 (+{PROJECT_URL})",
)

# Any of these means we have been told to stop, and the run ends. 429 is here by
# choice rather than necessity: it would normally be retried after a pause, but
# a scrape this small should never provoke one, so seeing it means an assumption
# is wrong and a human should look rather than the script pressing on.
BLOCKED_STATUS = {401, 403, 429}


class Blocked(RuntimeError):
    """The site refused us. Stop the run; do not retry under another identity."""


class Fetcher:
    """Sequential, rate-limited, archiving HTTP client.

    Sequential on purpose. Nine products is roughly 120 requests, which at one
    and a half seconds apart takes about three minutes. Concurrency would save
    two of those minutes and is not worth being an inconsiderate guest.
    """

    def __init__(
        self,
        raw_dir: Path,
        *,
        delay: float = 1.5,
        timeout: float = 20.0,
        max_attempts: int = 3,
    ) -> None:
        self.raw_dir = raw_dir
        self.delay = delay
        self.max_attempts = max_attempts
        self._last_request_at = 0.0
        self._client = httpx.Client(
            headers={
                "User-Agent": USER_AGENT,
                "Accept-Encoding": "gzip, deflate",
            },
            timeout=timeout,
            follow_redirects=True,
        )
        self.request_count = 0

    def __enter__(self) -> "Fetcher":
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.close()

    def _wait_turn(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.delay:
            time.sleep(self.delay - elapsed)
        self._last_request_at = time.monotonic()

    def get(
        self,
        url: str,
        *,
        archive_as: str,
        params: dict[str, Any] | None = None,
    ) -> httpx.Response:
        """Fetch one URL, archive the body, return the response.

        Retries a transient failure with a widening pause. Does not retry a
        refusal: if we are told no, the answer will not change on a second ask.
        """
        last_error: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            self._wait_turn()
            try:
                response = self._client.get(url, params=params)
            except httpx.RequestError as exc:
                last_error = exc
                log.warning("attempt %d/%d failed for %s: %s",
                            attempt, self.max_attempts, url, exc)
                if attempt < self.max_attempts:
                    time.sleep(self.delay * attempt)
                continue

            self.request_count += 1

            if response.status_code in BLOCKED_STATUS:
                raise Blocked(
                    f"{response.status_code} from {url}. Stopping the run. "
                    "Do not retry with a different user-agent."
                )
            if response.status_code >= 500:
                last_error = httpx.HTTPStatusError(
                    f"server error {response.status_code}",
                    request=response.request,
                    response=response,
                )
                log.warning("attempt %d/%d got %d for %s",
                            attempt, self.max_attempts, response.status_code, url)
                if attempt < self.max_attempts:
                    time.sleep(self.delay * attempt)
                continue

            response.raise_for_status()
            self._archive(archive_as, response.content)
            return response

        raise RuntimeError(f"giving up on {url} after {self.max_attempts} attempts") from last_error

    def get_json(self, url: str, *, archive_as: str,
                 params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fetch and decode JSON.

        Bazaarvoice answers HTTP 200 even when the request failed, reporting the
        problem only inside the body. Checking the status code alone would let a
        bad passkey through as an empty but apparently successful result, so the
        body is checked here for every JSON call.
        """
        payload = self.get(url, archive_as=archive_as, params=params).json()

        if payload.get("HasErrors"):
            errors = payload.get("Errors") or []
            detail = "; ".join(
                f"{e.get('Code')}: {e.get('Message')}" for e in errors
            ) or "unspecified"
            raise RuntimeError(f"{url} returned HTTP 200 but reported: {detail}")

        return payload

    def _archive(self, name: str, body: bytes) -> None:
        path = self.raw_dir / f"{name}.gz"
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wb") as fh:
            fh.write(body)
