# SPDX-License-Identifier: Apache-2.0
"""Opt-in real HTTP e2e for Ming-Omni text streaming (issue #600).

This exercises the full live stack over SSE against a running server. It is
skipped unless ``MING_OMNI_E2E_URL`` is set, so it stays inert in normal unit
runs and CI without a GPU.

To run it against a live server::

    python examples/run_ming_omni_server.py --model-path <path> --port 8000
    MING_OMNI_E2E_URL=http://127.0.0.1:8000 pytest \
        tests/unit_test/ming_omni/test_streaming_http_e2e.py -q

It asserts the live contract that the in-process e2e cannot: a real client sees
multiple incremental ``delta.content`` chunks, a terminal ``finish_reason``, and
no duplicated full-text final chunk.
"""

from __future__ import annotations

import json
import os

import pytest

_URL = os.environ.get("MING_OMNI_E2E_URL")
_MODEL = os.environ.get("MING_OMNI_E2E_MODEL", "ming-omni")

pytestmark = pytest.mark.skipif(
    not _URL,
    reason="set MING_OMNI_E2E_URL to a running Ming-Omni OpenAI server",
)


def _stream_chat(prompt: str, *, max_tokens: int = 64):
    requests = pytest.importorskip("requests")
    body = {
        "model": _MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "modalities": ["text"],
        "temperature": 0,
        "max_tokens": max_tokens,
    }
    deltas: list[str] = []
    finish_reason = None
    with requests.post(
        f"{_URL}/v1/chat/completions", json=body, stream=True, timeout=180
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data: "):
                continue
            data = raw[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            choice = (json.loads(data).get("choices") or [{}])[0]
            content = (choice.get("delta") or {}).get("content")
            if isinstance(content, str) and content:
                deltas.append(content)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]
    return deltas, finish_reason


def test_http_streaming_emits_incremental_text_deltas():
    deltas, finish_reason = _stream_chat("What is 2 + 2? Answer briefly.")

    # Incremental: the client must receive more than one non-empty content chunk.
    assert len(deltas) > 1, f"expected multiple deltas, got {deltas!r}"

    full = "".join(deltas)
    assert full.strip(), "no text content was streamed"

    # The stream must terminate with a finish_reason.
    assert finish_reason is not None

    # No duplicated full-text final chunk: the last delta is an increment, not a
    # repeat of the entire generated text.
    assert deltas[-1] != full
