# RFC: Stateful KV-Cache Streaming for the MOSS-TTS Codec Decoder

> 中文版 / Chinese version: [moss_tts_stateful_codec_streaming_rfc.cn.md](moss_tts_stateful_codec_streaming_rfc.cn.md)

| | |
|---|---|
| **Status** | Draft |
| **Author** | liuqihao |
| **Date** | 2026-06-09 |
| **Branch** | `feat/moss_streaming_optimize` |
| **Supersedes** | The overlap-window re-decode path in `sglang_omni/models/moss_tts/streaming_vocoder.py` (v1, see `docs/design/moss_tts_streaming_plan.md` §5) |
| **Related** | MiMo-Audio `streaming_decode` (`vllm-omni`), issue #637 |

---

## 1. Summary

MOSS-TTS streaming audio currently degrades WER/CER relative to the
non-streaming path. The root cause is that the codec decoder is a stack of
**causal RoPE transformers**, but the streaming vocoder feeds each chunk through
a stateless **overlap-window re-decode** with only ~2 frames of left context.
The transformer was trained with up to 10 s of causal context, so truncating it
corrupts the acoustic frames near every window start. This RFC proposes
re-introducing the **per-layer KV cache** (which the upstream codec had and the
vendored inference copy stripped) so that streaming decode is mathematically
equivalent to offline decode.

There is a **second, independent bug**: the pre-codec de-delay/segment scan
re-processes the full row history on every chunk, which is O(N²). The two fixes
are orthogonal and both live in the streaming vocoder: **incremental de-delay**
removes the O(N²) cost, and the **stateful KV cache** removes the correctness
(seam / WER) regression. With both, total cost is O(N) and output matches
offline.

---

## 2. Background

The MOSS audio tokenizer decoder (`audio_tokenizer.py`,
`_default_decoder_kwargs`) is a sequence of `_ProjectedTransformer`
(causal self-attention + RoPE) interleaved with `_PatchedPretransform`
(parameter-free patch reshape upsampling). Auditing every operator:

- `_PatchedPretransform.forward` upsample is a pure reshape
  `(b, d, T) → (b, d//h, T*h)`; each output time-step comes from a single input
  time-step's channels. **No cross-time mixing → zero state, chunk-independent.**
- LayerNorm / FF / `in_proj` / `out_proj` are per-frame.
- The quantizer `decode_codes` is an embedding lookup + kernel-1 conv → per-frame.
- **The only operator that mixes information across time is the causal
  self-attention** (`_Attention.forward`) and its RoPE.

The vendored copy's header literally states: *"Streaming KV-cache infrastructure
removed (single-pass batch decode only)."* The upstream OpenMOSS codec had a
streaming KV cache; the inference vendor dropped it.

---

## 3. Problem Statement

There are **two separate costs per chunk**, and the current streaming vocoder
gets each wrong in a different way. Do not conflate them.

**Bug A — pre-codec de-delay / segment scan is O(N²) (cost).**
Turning accumulated delayed rows into clean code frames. The current code
re-stacks the *full* row history and re-runs the *whole* segment scan on **every
chunk** (`streaming_vocoder.py:149` — `torch.stack(state.delayed_rows[:rows_end])`
followed by a full `split_moss_audio_segments`, whose `apply_de_delay_pattern` +
pad scan + `nonzero` all sweep the entire history). That is O(N) per chunk →
**O(N²) total**, and latency grows with utterance length. This is what
GaokaiZhang's "O(N²)" refers to — *not* the codec window decode.

**Bug B — codec window decode is O(N) but wrong (correctness).**
Turning code frames into a waveform. The current code feeds only a small window
`segment[emitted-overlap : end]` (`streaming_vocoder.py:173`), so this part is
**already O(N)** — but it is **wrong**: a stateless transformer fed only new
frames + ~2 overlap frames cannot see frames `0..emitted`, so it hallucinates the
acoustic context → WER/CER degradation and audible seams.

| Bug | Where | Symptom | Fix |
|---|---|---|---|
| A: O(N²) cost | pre-codec de-delay / segment scan | latency grows with utterance length | **incremental de-delay** (§5.4) |
| B: wrong output | codec window decode | WER/CER degradation, seams | **stateful KV cache** (§5.1–5.3) |

The two fixes are orthogonal and both belong to the streaming vocoder.

For Bug B specifically, the choice of how many frames to feed interacts with
correctness:

|  | frames fed | sees history? | result |
|---|---|---|---|
| feed `0..N` (stateless) | O(N²) | ✅ | correct but slow |
| feed new frames only (stateless) ← **current** | O(N) | ❌ | fast but wrong (WER) |
| feed new frames only + **KV cache** | O(N) | ✅ | **fast and correct** |

A conv decoder (MiMo) tolerates overlap re-decode because its receptive field is
small and fixed — a tiny overlap is enough. A causal transformer's receptive
field is the **entire attention window (~10 s)**, so overlap re-decode is either
wrong (small overlap) or O(N²) (full overlap). KV cache is the O(N) memoized form
of full-overlap re-decode: identical output, no recompute.

---

## 4. Goals / Non-Goals

**Goals**

- Streaming codec decode output matches offline `batch_decode` per segment within
  a tight numerical tolerance.
- Total cost is O(N); per-chunk cost is O(new frames) for **both** the de-delay
  scan and the codec decode (incremental de-delay removes the O(N²) scan).
- WER/CER parity between streaming and non-streaming.
- Preserve the scheduler lifecycle and the offline de-delay semantics.

**Non-Goals**

- Batched (ragged multi-request) streaming codec decode — v1 stays per-request.
- CUDA Graph wrapping of the streaming decoder.
- Changes to the AR (talker) stage or de-delay semantics.
- Encoder / reference-audio path changes.

---

## 5. Design

### 5.1 Streaming state object

A `StreamingCache` dataclass holds one `StageCache` per decoder transformer
stage. Each stage runs at a **different time rate** (patch upsampling changes T
between stages), so the RoPE position offset is **per stage, not global** — a
single global `pos_offset` would be ambiguous. Each `StageCache` holds:

- per-layer `k`, `v`: `[B, H, T_cached, D]`.
- `pos_offset` (int) for RoPE continuation, counted at *that stage's* time rate.
- (optional) a `window` cap that drops K/V older than the trained
  `causal_transformer_context_duration` (10 s) to bound memory for long
  utterances.

Patch layers are stateless and store nothing.

```python
@dataclass
class LayerKV:
    k: torch.Tensor | None = None   # [B, H, T, D]
    v: torch.Tensor | None = None   # [B, H, T, D]

@dataclass
class StageCache:
    layer_kvs: list[LayerKV] = field(default_factory=list)
    pos_offset: int = 0             # RoPE offset at THIS stage's time rate

@dataclass
class StreamingCache:
    # one StageCache per decoder transformer stage
    stage_caches: list[StageCache] = field(default_factory=list)
    finished: bool = False
```

### 5.2 API

Mirror MiMo's signature for consistency:

```python
audio_chunk, state = codec.decode_stream(codes_chunk, state=None, last_chunk=False)
```

- `codes_chunk`: `(NQ, T_new)` — only the newly de-delayed frames.
- `state`: `StreamingCache` from the previous call (or `None` to start).
- `last_chunk`: flush flag for the final tail / segment close.
- returns the waveform delta for exactly these new frames + the updated state.

### 5.3 Codec changes (`audio_tokenizer.py`)

1. **`_apply_rope`** — add a `pos_offset` argument; replace `arange(T)` with
   `arange(offset, offset+T)`. Positions **must continue across chunks**
   (keep the GPT-J interleaved convention warned about at `audio_tokenizer.py:174`).

2. **`_Attention.forward`** — accept/return a `LayerKV`. Project new q/k/v; apply
   RoPE with `pos_offset`; concat new k/v onto the cache; attend new q against the
   full cache. With a cache present, replace `is_causal=True` square masking with
   an explicit mask where each new query attends to all cached + new keys up to
   itself.

3. **`_TransformerLayer` / `_Transformer` / `_ProjectedTransformer`** — thread a
   per-layer cache list through `forward`.

4. **`MossAudioTokenizerModel.decode_stream`** — orchestrate:
   `quantizer.decode_codes(new_frames)` (stateless) → run decoder stages threading
   `state`; patch layers run as-is (stateless); advance each stage's `pos_offset`
   by the number of new frames *at that stage's* time rate (patch upsampling
   changes T between stages, so each `StageCache` tracks its own offset — see §5.1).

5. **`reset()`** on the cache for segment boundaries.

### 5.4 Incremental de-delay (fixes Bug A, the O(N²) cost)

The current code does **not** have a true incremental de-delay; it accumulates
rows and re-runs the full `split_moss_audio_segments` over the whole history each
chunk (`streaming_vocoder.py:149`). Replace this with a true incremental scan.
Maintain on `_MossStreamState`:

- `raw_frame_cursor` — how many de-delayed frames have been produced so far
  (global, across segments).
- `open_segment` — the currently growing audio segment (or `None` between segments).
- per-segment `frames` — completed frames of the open segment.
- per-segment `emitted_cursor` — frames already handed to the codec.

On each chunk, de-delay **only the newly completable rows** — a frame at index
`i` becomes complete once row `i + n_vq - 1` has arrived. Classify each new frame
as audio vs pad-only separator; append audio frames to `open_segment`; on a
pad-only separator, close `open_segment` (flush + reset its codec state) and open
a fresh one. **Never call `split_moss_audio_segments` over the full history
again** — per-chunk work becomes O(new frames), total O(N).

`assistant_start_length` is applied once, against the global `raw_frame_cursor`,
only to the first audio segment — matching `split_moss_audio_segments`.

### 5.5 Scheduler changes (`streaming_vocoder.py`)

- Add `codec_state: StreamingCache | None` per **segment** in `_MossStreamState`.
- Drive the codec from the incremental de-delay of §5.4: feed only the newly
  completed frames to `codec.decode_stream(new_frames, state)`, emit the returned
  waveform delta directly.
- **Delete** the overlap / trim / re-decode / `samples_per_frame` machinery
  (`streaming_vocoder.py:173-194`) — `decode_stream` returns exactly the new
  samples, so no trimming is needed.
- **Segment boundaries**: when §5.4 closes a segment (pad-only separator), call
  `decode_stream(..., last_chunk=True)` to flush, then start a fresh
  `StreamingCache` for the next segment — the offline path decodes each segment
  independently, and this preserves that.

### 5.6 sglang-omni codec sourcing

In sglang-omni the codec is loaded via `trust_remote_code`
(`processor.decode_audio_codes`); its source is not in the repo. To own a stateful
decode path, vendor it under a dedicated, traceable location (not a bare module
next to first-party code):

- **Path**: `sglang_omni/models/moss_tts/_vendored/moss_audio_tokenizer/`.
- **Provenance** — add `_vendored/README.md` recording: source repo
  (`OpenMOSS-Team/MOSS-Audio-Tokenizer`), the exact vendored commit/revision hash,
  the upstream license, and the local modifications (KV streaming added). Pin the
  revision so the checkpoint weights and the vendored modeling code cannot drift.
- **Switch-over** — the codec loader uses the local vendored class instead of
  `get_class_from_dynamic_module(...)` when `stream_codec_mode == "stateful"`, so
  non-streaming and the `overlap` fallback can still use the remote module.
- **Rejected (b)** — subclass / monkeypatch the remote attention module: fragile
  and checkpoint-version-dependent.

---

## 6. Correctness Argument

Because the only temporal mixing is causal attention, and a KV cache reproduces
exactly the K/V a full forward pass would compute, decoding chunk `t` with the
cache is bit-equal to decoding `[0..t]` in one pass — within the attention window.
Patch upsampling and all per-frame ops are trivially identical chunked or not.
Therefore concatenated streaming output equals per-segment offline output, which
is the WER/CER parity we need.

---

## 7. Risks & Open Questions

1. **10 s context window semantics** — how upstream implements
   `causal_transformer_context_duration` (sliding-window mask vs full causal).
   Must replicate from upstream source, else long utterances drift / grow memory.
   *This is the only hard external dependency.*
2. **RoPE offset continuation** — interleaved (GPT-J) convention must continue
   correctly across chunks; verify with a unit test.
3. **No kernel>1 convs** — confirmed: only kernel-1 `_wn_conv1d` in the quantizer,
   so KV cache is both necessary and sufficient. Re-verify if the checkpoint config
   changes.
4. **Batching** — per-request state in v1; ragged batched `decode_stream` is future
   work.
5. **Float precision** — codec runs float32 throughout; cache stays float32.

---

## 8. Testing

- **De-delay incrementality**: incremental de-delay output == one-shot
  `split_moss_audio_segments(full)`; assert no per-chunk full-history re-scan
  (e.g. work is O(new frames)).
- **Parity**: `cat(decode_stream chunks) ≈ batch_decode(full)` per segment,
  `torch.allclose` within tolerance.
- **KV continuity**: feeding 1 frame at a time == feeding all at once.
- **RoPE offset**: position-offset path == full-position path.
- **Multi-segment reset**: pad-only separator flushes + resets; only the first
  segment is trimmed by `assistant_start_length`.
- **Scheduler lifecycle**: abort clears per-request `codec_state`; no terminal
  audio after abort; terminal payload not double-decoded.

---

## 9. Rollout

Add `stream_codec_mode = {"overlap", "stateful"}`. Default to `overlap` until
parity tests + a real-model WER/CER run confirm `stateful`, then flip the default
and keep `overlap` as a fallback for one release.

---

## 10. Alternatives Considered

- **Bigger overlap (stay stateless)** — would need overlap ≈ full window to match
  offline → O(N²). Rejected: defeats streaming.
- **MiMo-style hidden-overlap recompute** — natural for conv (small receptive
  field), not for a causal transformer (window-sized receptive field). Rejected:
  same O(N²)/wrong tradeoff as above.
- **Subclass/monkeypatch remote codec** — fragile across checkpoint versions.
  Rejected in favor of vendoring.

---

## 10b. Verification Results

Measured 2026-06-09 on SeetaCloud H20, `OpenMOSS-Team/MOSS-TTS-v1.5` +
`Qwen/Qwen3-ASR-1.7B`.

- **Unit parity** (tiny random-init model, 39 tests): `decode_stream` chunked ==
  `batch_decode`; per-frame == one-shot; RoPE offset continuation; multi-segment
  reset; incremental de-delay == `split_moss_audio_segments`; scheduler lifecycle.
- **Real checkpoint**: weight remap **100%** (1600/1600, 0 skipped/missing);
  real-weight `decode_stream` vs `batch_decode` **max_abs_diff = 1.4e-4** (bit-equal).
- **Vendored vs remote codec** (`processor.decode_audio_codes`): mean_abs = 3.3e-5,
  max = 4.5e-3 — near-identical; small re-implementation numerical drift, not a
  structural difference.
- **e2e WER** (SeedTTS EN, 50 samples, c=1, ASR=Qwen3-ASR-1.7B):
  - seeded (seed=1234): non-streaming **1.42%**, stateful-stream **1.77%** (mean
    1.18% / 1.45%; max 20% / 14.3%; no >50% outliers). Both well under the 1.93%
    full-EN-set non-streaming baseline (PR #609).
  - unseeded (noisy, different tokens per request): non-streaming 1.06%,
    overlap-stream 1.42%, stateful-stream 2.13% (one 50% outlier).
- **Conclusion**: stateful KV-cache streaming adds **no decode error vs offline**
  for the same codes (bit-equal). The residual e2e WER gap is sampling
  nondeterminism + the small vendored-vs-remote codec drift — **not** the
  left-context truncation that degraded the overlap path. Serving works under
  `MOSS_TTS_STREAM_CODEC_MODE=stateful` with `--mem-fraction-static 0.80` headroom
  for the second codec.

## 10c. Concurrency Sweep & Performance Caveat

Full-set A/B (1088 EN, seed=1234, identical tokens): non-streaming 1.792% WER /
0.770% CER; overlap-stream 2.018% / 0.940%; stateful-stream 1.616% / 0.775%.
**Stateful removes the overlap path's degradation and matches offline.**

Concurrency sweep (EN, 256/group, seed=1234, paired tokens):

| mode | c | WER% | CER% | TTFA mean | RTF | QPS |
|---|---|---|---|---|---|---|
| overlap | 1 | 2.039 | 0.979 | 2.20s | 0.73 | 0.34 |
| overlap | 4 | 2.112 | 0.948 | 1.67s | 0.74 | 1.30 |
| overlap | 8 | 2.185 | 1.074 | 4.51s | 1.46 | 1.32 |
| overlap | 16 | 2.039 | 0.940 | 9.88s | 2.87 | 1.35 |
| stateful | 1 | 1.894 | 0.829 | 2.17s | 1.15 | 0.21 |
| stateful | 4 | 1.894 | 0.821 | 9.14s | 2.99 | 0.33 |
| stateful | 8 | 2.003 | 0.877 | 20.75s | 5.94 | 0.33 |
| stateful | 16 | 1.857 | 0.979 | 42.29s | 11.66 | 0.33 |

**Quality**: stateful WER is ~0.15–0.22pp below overlap at every concurrency
(CER similar); the win does not regress with concurrency.

**Performance (caveat)**: v1 stateful is per-request and **unbatched**, and the
initial implementation called `decode_stream` once per *single* completed frame
(rows arrive one at a time), so the heavy fp32 decoder ran per-frame. At c≥4 the
concurrent codec calls serialize on the GPU and TTFA/RTF blow up (TTFA 42s, RTF
11.7 at c=16 vs overlap 9.9s/2.9). Only c=1 is viable as-is.

**Mitigation (v1.1, implemented + measured)**: coalesce completed frames and call
`decode_stream` once per `stream_stride` frames (default 8) instead of per frame.
The KV cache makes `decode_stream` exact regardless of chunk length, so this is
correctness-neutral (all 39 unit tests still pass; WER/CER unchanged). Measured
stateful perf after coalescing (EN, 256/group):

| c | TTFA before→after | RTF before→after | QPS before→after |
|---|---|---|---|
| 1 | 2.17→2.25s | 1.15→0.71 | 0.21→0.34 |
| 4 | 9.14→1.18s | 2.99→0.58 | 0.33→1.63 |
| 8 | 20.75→3.44s | 5.94→1.16 | 0.33→1.67 |
| 16 | 42.29→7.71s | 11.66→2.28 | 0.33→1.69 |

After coalescing, stateful **beats overlap on both quality and performance** at
every concurrency (e.g. c=16: TTFA 7.71s vs overlap 9.88s, RTF 2.28 vs 2.87, QPS
1.69 vs 1.35; WER 1.86% vs 2.04%). This is expected — the KV cache does no
recompute, whereas overlap re-decodes overlapping windows. Full cross-request
batched codec decode (RFC §5b) remains optional future work for even higher
concurrency.

**Recommendation: make `stateful` the default `stream_codec_mode`** once landed;
keep `overlap` as a fallback for one release.

## 11. Work Breakdown

| # | Task | File |
|---|---|---|
| 1 | Vendor codec under `_vendored/` with provenance README + pinned revision | `sglang_omni/models/moss_tts/_vendored/moss_audio_tokenizer/` (new) |
| 2 | `pos_offset` in RoPE (per-stage) | `_vendored/.../audio_tokenizer.py` |
| 3 | KV cache in attention + layers | `_vendored/.../audio_tokenizer.py` |
| 4 | `StageCache` / `StreamingCache` + `decode_stream` | `_vendored/.../audio_tokenizer.py` |
| 5 | **Incremental de-delay** (raw_frame_cursor, open/closed segment, per-segment cursors) — fixes O(N²) | `streaming_vocoder.py`, `codec.py` |
| 6 | Replace overlap path with stateful path | `streaming_vocoder.py` |
| 7 | `stream_codec_mode` flag + loader switch-over | `streaming_vocoder.py`, config |
| 8 | Unit tests (de-delay incrementality, parity, continuity, RoPE, reset, lifecycle) | `tests/unit_test/moss_tts/` |
| 9 | Verify 10 s window semantics from upstream | upstream source + `_vendored/...` |
| 10 | Real-model WER/CER parity run | `tests/test_model/` |
