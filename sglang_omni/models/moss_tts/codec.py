# SPDX-License-Identifier: Apache-2.0
"""Delay-pattern helpers for MOSS-TTS audio codes."""

from __future__ import annotations

from typing import Any

import torch


def apply_de_delay_pattern(delayed_codes: torch.Tensor) -> torch.Tensor:
    """Convert delayed RVQ rows back to normal ``[time, n_vq]`` codes."""

    if delayed_codes.ndim != 2:
        raise ValueError("MOSS-TTS delayed audio codes must be rank-2")
    if delayed_codes.shape[0] < delayed_codes.shape[1]:
        return delayed_codes.new_empty((0, delayed_codes.shape[1]))

    tokens = delayed_codes.new_full(
        (
            delayed_codes.shape[0] - delayed_codes.shape[1] + 1,
            delayed_codes.shape[1],
        ),
        0,
    )
    for idx in range(delayed_codes.shape[1]):
        tokens[:, idx] = delayed_codes[idx : idx + tokens.shape[0], idx]
    return tokens


def split_moss_audio_segments(
    delayed_audio_codes: Any,
    *,
    audio_pad_code: int,
    assistant_start_length: int = 0,
) -> list[torch.Tensor]:
    """Extract contiguous decoded audio-code segments from delayed rows."""

    if delayed_audio_codes is None:
        return []
    if not isinstance(delayed_audio_codes, torch.Tensor):
        delayed_audio_codes = torch.as_tensor(delayed_audio_codes, dtype=torch.long)
    delayed_audio_codes = delayed_audio_codes.to(dtype=torch.long)
    if delayed_audio_codes.numel() == 0:
        return []

    audio_codes = apply_de_delay_pattern(delayed_audio_codes)
    if audio_codes.numel() == 0:
        return []

    pad_code = int(audio_pad_code)
    is_pad = (audio_codes == pad_code).all(dim=1)
    is_complete_code = ((audio_codes >= 0) & (audio_codes < pad_code)).all(dim=1)
    non_pad = (~is_pad) & is_complete_code
    if not bool(non_pad.any()):
        return []

    idx = torch.nonzero(non_pad, as_tuple=False).squeeze(1)
    break_points = torch.where(idx[1:] != idx[:-1] + 1)[0] + 1
    if break_points.numel() == 0:
        segments = [idx]
    else:
        segments = list(torch.tensor_split(idx, break_points.cpu().tolist()))

    code_segments = [audio_codes[segment].contiguous() for segment in segments]
    if assistant_start_length > 0 and code_segments:
        trim = min(int(assistant_start_length), int(code_segments[0].shape[0]))
        code_segments[0] = code_segments[0][trim:]
        code_segments = [segment for segment in code_segments if segment.numel() > 0]
    return code_segments


class IncrementalDeDelay:
    """Streaming de-delay that emits only newly-completed frames.

    Equivalent to running :func:`split_moss_audio_segments` over the full row
    history, but each :meth:`push` does **O(new rows)** work instead of
    re-stacking and re-scanning the whole history on every chunk (which is the
    O(N²) cost called out in the RFC §3 / §5.4).

    Frame ``f`` (0-indexed, in de-delayed space) is complete once row
    ``f + n_vq - 1`` has arrived; its codes are ``rows[f + idx][idx]`` for
    ``idx in 0..n_vq-1`` — the same diagonal read as
    :func:`apply_de_delay_pattern`.

    :meth:`push` and :meth:`finalize` return a flat list of events:
      * ``("frame", tensor[n_vq])`` — a complete audio frame of the open segment.
      * ``("close",)`` — the open segment ended (pad-only / incomplete separator).

    Segment boundaries match the offline path: contiguous non-pad complete frames
    form a segment; a pad-only or out-of-range frame closes the open segment.
    ``assistant_start_length`` trims the first audio frames of the **first** audio
    segment only, matching :func:`split_moss_audio_segments`.
    """

    def __init__(
        self,
        *,
        n_vq: int,
        audio_pad_code: int,
        assistant_start_length: int = 0,
    ) -> None:
        if n_vq <= 0:
            raise ValueError(f"n_vq must be > 0, got {n_vq}")
        self.n_vq = int(n_vq)
        self.pad = int(audio_pad_code)
        self.assistant_start_length = int(assistant_start_length)
        self._rows: list[torch.Tensor] = []
        self._frame_cursor = 0          # next frame index to attempt
        self._open = False              # is a segment currently open
        self._audio_segment_count = 0   # number of audio segments started
        self._first_trimmed = 0         # frames trimmed from the first segment

    @property
    def frame_cursor(self) -> int:
        return self._frame_cursor

    def push(self, new_rows: torch.Tensor | list[torch.Tensor]) -> list[tuple]:
        """Append new delayed rows and return events for newly-completed frames."""
        if isinstance(new_rows, torch.Tensor):
            if new_rows.ndim == 1:
                new_rows = [new_rows]
            else:
                new_rows = [row for row in new_rows]
        for row in new_rows:
            row = torch.as_tensor(row, dtype=torch.long).reshape(-1)
            if int(row.shape[0]) != self.n_vq:
                raise ValueError(
                    f"delayed row has {int(row.shape[0])} codebooks, expected {self.n_vq}"
                )
            self._rows.append(row)

        events: list[tuple] = []
        while self._frame_cursor + self.n_vq <= len(self._rows):
            f = self._frame_cursor
            frame = torch.stack(
                [self._rows[f + idx][idx] for idx in range(self.n_vq)]
            ).to(torch.long)
            events.extend(self._route(frame))
            self._frame_cursor += 1
        return events

    def finalize(self) -> list[tuple]:
        """Close any open segment at end-of-stream."""
        if self._open:
            self._open = False
            return [("close",)]
        return []

    def _route(self, frame: torch.Tensor) -> list[tuple]:
        is_pad = bool((frame == self.pad).all())
        is_complete = bool(((frame >= 0) & (frame < self.pad)).all())
        non_pad = (not is_pad) and is_complete
        if not non_pad:
            if self._open:
                self._open = False
                return [("close",)]
            return []
        if not self._open:
            self._open = True
            self._audio_segment_count += 1
        # Trim the first audio segment by assistant_start_length.
        if (
            self._audio_segment_count == 1
            and self._first_trimmed < self.assistant_start_length
        ):
            self._first_trimmed += 1
            return []
        return [("frame", frame)]
