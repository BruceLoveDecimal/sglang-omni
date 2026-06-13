# SPDX-License-Identifier: Apache-2.0
"""IncrementalDeDelay parity against the offline split_moss_audio_segments."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from sglang_omni.models.moss_tts.codec import (  # noqa: E402
    IncrementalDeDelay,
    split_moss_audio_segments,
)

PAD = 1024
NVQ = 4


def _build_delayed(tokens: torch.Tensor) -> torch.Tensor:
    """Inverse of apply_de_delay_pattern: clean ``[T, nq]`` → delayed rows."""
    t, nq = tokens.shape
    delayed = torch.full((t + nq - 1, nq), PAD, dtype=torch.long)
    for idx in range(nq):
        delayed[idx : idx + t, idx] = tokens[:, idx]
    return delayed


def _segments_from_events(events: list[tuple]) -> list[torch.Tensor]:
    segs: list[torch.Tensor] = []
    cur: list[torch.Tensor] = []
    for ev in events:
        if ev[0] == "frame":
            cur.append(ev[1])
        elif ev[0] == "close":
            if cur:
                segs.append(torch.stack(cur))
            cur = []
    if cur:
        segs.append(torch.stack(cur))
    return segs


def _run_incremental(delayed, chunk_sizes, assistant_start_length=0):
    dd = IncrementalDeDelay(
        n_vq=NVQ, audio_pad_code=PAD, assistant_start_length=assistant_start_length
    )
    events: list[tuple] = []
    pos = 0
    for cs in chunk_sizes:
        events.extend(dd.push(delayed[pos : pos + cs]))
        pos += cs
    events.extend(dd.push(delayed[pos:]))
    events.extend(dd.finalize())
    return _segments_from_events(events)


def _make_tokens(seed: int, pad_positions: list[int], t: int = 24) -> torch.Tensor:
    torch.manual_seed(seed)
    tokens = torch.randint(0, PAD, (t, NVQ), dtype=torch.long)
    for p in pad_positions:
        tokens[p, :] = PAD  # all-pad separator frame
    return tokens


def _assert_same(a: list[torch.Tensor], b: list[torch.Tensor]):
    assert len(a) == len(b), f"segment count {len(a)} != {len(b)}"
    for x, y in zip(a, b):
        assert x.shape == y.shape, f"{tuple(x.shape)} != {tuple(y.shape)}"
        assert torch.equal(x, y)


@pytest.mark.parametrize(
    "pad_positions",
    [[], [10], [8, 9], [0], [23], [5, 12, 18]],
)
@pytest.mark.parametrize("chunk", [1, 3, 5, 100])
def test_incremental_matches_offline(pad_positions, chunk):
    tokens = _make_tokens(seed=7, pad_positions=pad_positions)
    delayed = _build_delayed(tokens)
    offline = split_moss_audio_segments(delayed, audio_pad_code=PAD)
    chunk_sizes = [chunk] * (len(delayed) // chunk)
    inc = _run_incremental(delayed, chunk_sizes)
    _assert_same(inc, offline)


@pytest.mark.parametrize("asl", [0, 1, 5, 100])
def test_assistant_start_length_trims_first_segment_only(asl):
    tokens = _make_tokens(seed=11, pad_positions=[10])
    delayed = _build_delayed(tokens)
    offline = split_moss_audio_segments(
        delayed, audio_pad_code=PAD, assistant_start_length=asl
    )
    inc = _run_incremental(delayed, [1] * len(delayed), assistant_start_length=asl)
    _assert_same(inc, offline)
