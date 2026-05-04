"""HTTP client to the Java backend's stateless /render endpoint.

We use the stateless endpoint (Stage 1), not the persisted one (Stage 3), because
the dataset workflow doesn't need DB rows for every example — labels live in our
own parquet file. This also keeps the backend free to scale independently of the
ML side.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class RenderedSample:
    image_bytes: bytes
    width_px: int
    height_px: int
    perf: dict[str, int]
    request_id: str


class BackendError(RuntimeError):
    pass


class BackendClient:
    """Thin wrapper around `POST /render` with retry on transient failures.

    The retry budget covers the typical interruptions during a long dataset run
    on a laptop: backend GC pauses, the OS pre-empting the JVM for ImageIO, a
    transient `EAGAIN` on socket write. It does not cover 4xx — those are coding
    bugs and should fail loudly.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 0.5,
    ) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._max_retries = max_retries
        self._backoff = retry_backoff_seconds

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> BackendClient:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def render(self, recipe: dict[str, Any]) -> RenderedSample:
        body = {"recipe": recipe}
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = self._client.post("/render", json=body)
                # 4xx = caller's fault, do not retry.
                if 400 <= resp.status_code < 500:
                    raise BackendError(
                        f"backend rejected recipe (HTTP {resp.status_code}): {resp.text[:500]}"
                    )
                resp.raise_for_status()
                payload = resp.json()
                return RenderedSample(
                    image_bytes=base64.b64decode(payload["imageBase64"]),
                    width_px=int(payload["widthPx"]),
                    height_px=int(payload["heightPx"]),
                    perf=dict(payload["performance"]),
                    request_id=str(payload["requestId"]),
                )
            except (httpx.TransportError, httpx.HTTPStatusError) as ex:
                last_error = ex
                if attempt >= self._max_retries:
                    break
                # Exponential-ish backoff. Production would jitter; for a serial
                # dataset run jitter is overkill.
                time.sleep(self._backoff * (2**attempt))
        raise BackendError(
            f"render failed after {self._max_retries} retries: {last_error}"
        ) from last_error
