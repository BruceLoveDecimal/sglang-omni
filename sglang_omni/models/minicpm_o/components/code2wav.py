# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens chunk by chunk with a cached speaker reference."""

from __future__ import annotations

import io
import os
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.flow import (
    FlowCache,
    FlowCacheRow,
    flow_cache_row,
    stack_flow_caches,
    trim_flow_cache,
)
from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

FLOW_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

OUTPUT_SAMPLE_RATE = 24000
CODEC_TOKEN_RATE = 25
SAMPLES_PER_CODEC_TOKEN = OUTPUT_SAMPLE_RATE // CODEC_TOKEN_RATE
MEL_FRAME_RATE = 50
SAMPLES_PER_MEL_FRAME = OUTPUT_SAMPLE_RATE // MEL_FRAME_RATE

# HiFT is not causal: every chunk re-vocodes the previous chunk's last frames,
# and the overlapping waveform is cross-faded instead of being emitted twice.
MEL_OVERLAP_FRAMES = 8
SOURCE_OVERLAP_SAMPLES = MEL_OVERLAP_FRAMES * SAMPLES_PER_MEL_FRAME
# Flow attention keeps the whole reference plus this many recent frames.
FLOW_CACHE_TAIL_FRAMES = 100
# Each entry holds attention state for every denoising step of one reference.
FLOW_PROMPT_CACHE_CAPACITY = 4
SPEAKER_PROMPT_CACHE_CAPACITY = 32


@dataclass(kw_only=True)
class RowState:
    """Decoding progress of one request across chunks."""

    tokens: torch.Tensor
    chunks: list[tuple[int, int, bool]]
    flow: FlowCacheRow
    mel_overlap: torch.Tensor
    source_overlap: torch.Tensor
    speech_overlap: torch.Tensor
    pieces: list[torch.Tensor] = field(default_factory=list)


def plan_chunks(
    num_tokens: int, chunk_tokens: int, lookahead: int
) -> list[tuple[int, int, bool]]:
    """Split num_tokens into (start, end, last) chunks with lookahead for all but the last."""
    chunks = []
    start = 0
    while num_tokens - start > chunk_tokens + lookahead:
        chunks.append((start, start + chunk_tokens, False))
        start += chunk_tokens
    chunks.append((start, num_tokens, True))
    return chunks


class MiniCPMOCode2Wav(nn.Module):
    """Convert codec tokens into a float32 waveform with Token2wav."""

    def __init__(
        self,
        model_path: str,
        *,
        chunk_tokens: int,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        n_timesteps: int = 10,
        prompt_wav: str | None = None,
    ) -> None:
        super().__init__()
        from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"Token2wav requires a CUDA device, got {device}")
        self.device_context = torch.cuda.device(dev.index or 0)

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        if dtype is None:
            torch_dtype = torch.float32
        elif isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            torch_dtype = resolve_dtype(dtype)
        if torch_dtype not in FLOW_DTYPES:
            raise ValueError(
                f"Code2Wav dtype must be float32, float16, or bfloat16, got {dtype}"
            )
        with self.device_context:
            self.token2wav = Token2Wav(
                Path(asset_dir), device=dev, dtype=torch_dtype, n_timesteps=n_timesteps
            )
        min_chunk_tokens = -(-MEL_OVERLAP_FRAMES // self.token2wav.flow.up_rate)
        if chunk_tokens < min_chunk_tokens:
            raise ValueError(
                f"chunk_tokens must be at least {min_chunk_tokens}, got {chunk_tokens}"
            )
        self.chunk_tokens = chunk_tokens

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        self.default_prompt_wav = prompt_wav
        # Keyed by reference so a mixed-reference batch never thrashes one slot.
        self.prompt_cache: OrderedDict[str, tuple] = OrderedDict()
        self.prompt_cache_capacity = SPEAKER_PROMPT_CACHE_CAPACITY
        self.flow_cache: OrderedDict[str, FlowCache] = OrderedDict()
        self.flow_cache_capacity = FLOW_PROMPT_CACHE_CAPACITY
        self.speech_window = torch.hamming_window(
            2 * SOURCE_OVERLAP_SAMPLES, periodic=False, device=dev
        )
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        prompt_wav: str | bytes | None = None,
        **_: object,
    ) -> dict[str, object]:
        """Vocode EOS-stripped codec tokens using the supplied or default reference."""
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            waveform = np.zeros(0, dtype=np.float32)
        else:
            with self.device_context:
                reference = self.resolve_prompt_wav(prompt_wav)
                waveform = self.vocode([tokens], reference)[0]
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def resolve_prompt_wav(self, prompt_wav: str | bytes | None) -> str | bytes:
        if prompt_wav is not None:
            resolved = prompt_wav
        elif self.default_prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            resolved = self.default_prompt_wav
        return resolved

    @staticmethod
    def prompt_key(prompt_wav: str | bytes) -> str:
        if isinstance(prompt_wav, bytes):
            return f"bytes:{hash_bytes(prompt_wav)}"
        return reference_path_cache_key(prompt_wav) or f"path:{prompt_wav}"

    def speaker_prompt(
        self, prompt_wav: str | bytes | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_wav = self.resolve_prompt_wav(prompt_wav)
        prompt_key = self.prompt_key(prompt_wav)
        cached = self.prompt_cache.get(prompt_key)
        if cached is not None:
            self.prompt_cache.move_to_end(prompt_key)
            return cached
        # Bytes references decode in memory; they are not spilled to a temp file.
        source = io.BytesIO(prompt_wav) if isinstance(prompt_wav, bytes) else prompt_wav
        prompt = self.token2wav.prepare_prompt(source)
        self.prompt_cache[prompt_key] = prompt
        if len(self.prompt_cache) > self.prompt_cache_capacity:
            self.prompt_cache.popitem(last=False)
        return prompt

    def flow_autocast(self) -> torch.autocast:
        return torch.amp.autocast(
            self.token2wav.device.type,
            dtype=self.token2wav.dtype,
            enabled=self.token2wav.dtype != torch.float32,
        )

    def flow_prompt_cache(self, prompt_wav: str | bytes | None) -> FlowCache:
        """Flow state after consuming a reference, shared by every request using it."""
        prompt_wav = self.resolve_prompt_wav(prompt_wav)
        prompt_key = self.prompt_key(prompt_wav)
        cached = self.flow_cache.get(prompt_key)
        if cached is not None:
            self.flow_cache.move_to_end(prompt_key)
            return cached
        prompt_tokens, _, speaker_embedding, prompt_mels = self.speaker_prompt(
            prompt_wav
        )
        with self.flow_autocast():
            cache = self.token2wav.flow.prompt_cache(
                prompt_tokens,
                prompt_mels,
                speaker_embedding,
                self.token2wav.n_timesteps,
            )
        self.flow_cache[prompt_key] = cache
        if len(self.flow_cache) > self.flow_cache_capacity:
            self.flow_cache.popitem(last=False)
        return cache

    @torch.inference_mode()
    def vocode(
        self,
        token_sequences: Sequence[Sequence[int]],
        prompt_wav: str | bytes | Sequence[str | bytes] | None = None,
    ) -> list[np.ndarray]:
        """Decode rows chunk by chunk, batching chunks of equal shape across rows."""
        if not token_sequences:
            return []
        if any(len(tokens) == 0 for tokens in token_sequences):
            raise ValueError("codec token sequences must be non-empty")

        batch_size = len(token_sequences)
        if isinstance(prompt_wav, (list, tuple)):
            if len(prompt_wav) != batch_size:
                raise ValueError(
                    f"prompt_wav count {len(prompt_wav)} does not match "
                    f"token sequence count {batch_size}"
                )
            references = list(prompt_wav)
        else:
            references = [prompt_wav] * batch_size

        flow = self.token2wav.flow
        hift = self.token2wav.hift
        device = self.token2wav.device
        lookahead = flow.pre_lookahead_len
        up_rate = flow.up_rate
        tail_tokens = FLOW_CACHE_TAIL_FRAMES // up_rate
        speaker_embeddings = torch.cat(
            [self.speaker_prompt(reference)[2] for reference in references], dim=0
        )
        rows = [
            RowState(
                tokens=torch.tensor(tokens, dtype=torch.int32, device=device),
                chunks=plan_chunks(len(tokens), self.chunk_tokens, lookahead),
                flow=FlowCacheRow(batch=self.flow_prompt_cache(reference), index=0),
                mel_overlap=torch.zeros(flow.output_size, 0, device=device),
                source_overlap=torch.zeros(1, 0, device=device),
                speech_overlap=torch.zeros(0, device=device),
            )
            for tokens, reference in zip(token_sequences, references, strict=True)
        ]

        # All active rows are at the same chunk index, so grouping by chunk
        # width and finality also aligns the HiFT overlap windows.
        for step in range(max(len(row.chunks) for row in rows)):
            groups: dict[tuple[int, bool], list[int]] = {}
            for idx, row in enumerate(rows):
                if step < len(row.chunks):
                    start, end, last = row.chunks[step]
                    groups.setdefault((end - start, last), []).append(idx)
            for (_, last), indices in groups.items():
                members = [rows[idx] for idx in indices]
                token_end = 0 if last else lookahead
                tokens = torch.stack(
                    [
                        row.tokens[
                            row.chunks[step][0] : row.chunks[step][1] + token_end
                        ]
                        for row in members
                    ]
                )
                with self.flow_autocast():
                    mel, flow_cache = flow.decode_chunk(
                        tokens,
                        speaker_embeddings[indices],
                        stack_flow_caches([row.flow for row in members], up_rate),
                        last_chunk=last,
                        n_timesteps=self.token2wav.n_timesteps,
                    )
                if (
                    len(set(flow_cache.token_lens)) == 1
                    and len(set(flow_cache.prompt_tokens)) == 1
                ):
                    flow_cache = trim_flow_cache(
                        flow_cache, tail_tokens=tail_tokens, up_rate=up_rate
                    )
                    for row_idx, row in enumerate(members):
                        row.flow = FlowCacheRow(batch=flow_cache, index=row_idx)
                else:
                    for row_idx, row in enumerate(members):
                        row.flow = FlowCacheRow(
                            batch=trim_flow_cache(
                                flow_cache_row(flow_cache, row_idx, up_rate),
                                tail_tokens=tail_tokens,
                                up_rate=up_rate,
                            ),
                            index=0,
                        )

                mel = torch.cat(
                    [torch.stack([row.mel_overlap for row in members]), mel.float()],
                    dim=2,
                )
                source_overlap = torch.stack([row.source_overlap for row in members])
                speech, source = hift(
                    mel, source_overlap if source_overlap.shape[2] else None
                )
                speech_overlap = torch.stack([row.speech_overlap for row in members])
                if speech_overlap.shape[1]:
                    fade_len = SOURCE_OVERLAP_SAMPLES
                    speech[:, :fade_len] = (
                        speech[:, :fade_len] * self.speech_window[:fade_len]
                        + speech_overlap * self.speech_window[fade_len:]
                    )
                for row_idx, row in enumerate(members):
                    if last:
                        row.pieces.append(speech[row_idx])
                    else:
                        row.pieces.append(speech[row_idx, :-SOURCE_OVERLAP_SAMPLES])
                        row.mel_overlap = mel[row_idx, :, -MEL_OVERLAP_FRAMES:]
                        row.source_overlap = source[
                            row_idx, :, -SOURCE_OVERLAP_SAMPLES:
                        ]
                        row.speech_overlap = speech[row_idx, -SOURCE_OVERLAP_SAMPLES:]

        waveforms = pad_sequence(
            [torch.cat(row.pieces) for row in rows], batch_first=True
        ).cpu()
        outputs = []
        for idx, row in enumerate(rows):
            samples = row.tokens.numel() * SAMPLES_PER_CODEC_TOKEN
            assert sum(piece.numel() for piece in row.pieces) == samples
            outputs.append(waveforms[idx, :samples].numpy())
        return outputs
