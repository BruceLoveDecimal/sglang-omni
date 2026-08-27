# Qwen3-TTS stateful incremental Codec decoding — T-PR8

Spec for **T-PR8** of the Qwen3-TTS optimization roadmap
([#1754](https://github.com/sgl-project/sglang-omni/issues/1754)): integrate
stateful incremental Codec decoding into `Qwen3TTSStreamingVocoderScheduler`
with slot-indexed batched state management.

**This PR stacks on T-PR7**, which is already implemented as two draft PRs:

* [#1756](https://github.com/sgl-project/sglang-omni/pull/1756) — the
  incremental decoder core, `sglang_omni/models/qwen3_tts/incremental_codec.py`.
* [#1757](https://github.com/sgl-project/sglang-omni/pull/1757) — stacked on
  #1756; the opt-in B=1 eager serving path behind
  `enable_stateful_codec_decoder`.

T-PR8 adds no new decoder math. It changes *where the state lives* and *how many
streams execute per launch*, and it reconnects the incremental path to the async
vocoder machinery that #1757 deliberately switches off.

## Problem

Every streaming decode on the shipped path rebuilds a left-context window and
reruns the whole speech-tokenizer decoder over it. `_build_decode_plan` slices

```
[absolute_emitted - stream_left_context_frames, ref_frames + generated_frames)
```

and `_extract_delta` throws away the leading `left_context * total_upsample`
samples. With the current defaults (`DEFAULT_QWEN3_TTS_LEFT_CONTEXT_FRAMES = 16`,
`DEFAULT_QWEN3_TTS_STREAM_FOLLOWUP_STRIDE = 8`) the steady state decodes 24
frames to publish 8 — **3x redundant Codec work per emitted chunk**, and the
redundancy grows as the follow-up stride shrinks, which is exactly the direction
T-PR6 wants to move for TTFA.

There is a second, quieter cost. The decoder's `pre_transformer` uses sliding
window attention with `config.sliding_window = 72`, but a follow-up decode only
ever sees 24 frames. So the shipped streaming output is **already an
approximation** of full-sequence decoding: the Transformer context is truncated
to a third of the window the model was trained with. Stateful decoding removes
that truncation as a side effect, so the incremental path is expected to be
*closer* to the full-sequence reference than the current streaming path, not
merely equal to it. Accuracy reports must therefore carry three rows —
full-sequence, current streaming, incremental streaming — or the difference gets
misread as a regression.

## Decoder structure and state inventory

`Qwen3TTSTokenizerV2Decoder.forward` is

```
quantizer -> pre_conv -> pre_transformer -> upsample[] -> decoder[]
```

Checkpoint config (`Qwen3TTSTokenizerV2DecoderConfig` defaults, 12.5 Hz frames,
`total_upsample = prod((8,5,4,3) + (2,2)) = 1920`):

| field | value |
| :--- | :--- |
| `num_hidden_layers` | 8 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 16 |
| `hidden_size` / `latent_dim` | 1024 / 1024 |
| `sliding_window` | 72 |
| `num_quantizers` | 16 |
| `upsampling_ratios` / `upsample_rates` | (2, 2) / (8, 5, 4, 3) |
| `decoder_dim` | 1536 |

Per-module state requirement:

| module | shape parameters | incremental state | size |
| :--- | :--- | :--- | :--- |
| `quantizer` (split RVQ) | per-frame codebook lookup | none | — |
| `pre_conv` | CausalConv1d k=3 s=1 | input history | 2 frames |
| `pre_transformer` | sliding-window attn, 8 layers | per-layer K/V + absolute frame position | <=71 frames |
| `upsample[i][0]` | ConvTranspose1d k=s=`ratio` | **none** (`right_pad == 0`) | — |
| `upsample[i][1]` | ConvNeXt, dwconv k=7 | input history | 6 steps |
| `decoder[0]` | CausalConv1d k=7 | input history | 6 steps |
| `DecoderBlock[i]` head | ConvTranspose1d k=2r s=r | output overlap | r samples |
| `DecoderBlock[i]` residuals | CausalConv1d k=7, dilation 1/3/9 | input history | 6 / 18 / 54 steps |
| `decoder[-1]` | CausalConv1d k=7 | input history | 6 steps |
| `SnakeBeta`, `LayerNorm`, `RMSNorm`, MLPs | elementwise / per-position | none | — |

Four structural properties make bit-faithful incremental decoding possible,
rather than an approximation that would need a quality argument:

1. **Zero-initialized state is an exact cold start.** Every causal conv in this
   decoder left-pads with `F.pad(..., value=0)`, so zeroed history buffers are
   identical to what the full-sequence path sees at `t = 0`. Cold start needs no
   special case and no warm-up discard.
2. **ConvTranspose overlap is bounded by one input step.** In `DecoderBlock` the
   transposed conv has `kernel_size = 2r`, `stride = r`, and the module trims
   `right_pad = r`. Output block `[j*r, (j+1)*r)` depends only on input steps `j`
   and `j-1`, so a single `r`-sample overlap buffer is sufficient and exact. The
   `upsample` transposed convs have `k == s`, produce no overlap, and are
   stateless.
3. **RoPE uses absolute positions.** Keys are rotated by absolute position before
   entering the cache, so evicting keys outside the sliding window is exact,
   provided the state carries the absolute frame position and the per-step mask
   reproduces `create_sliding_window_causal_mask` semantics.
4. **`extra_padding` is always zero here.** For `stride == 1`,
   `_get_extra_padding_for_conv1d` reduces to `0` identically, and every causal
   conv in the decode path uses the default `stride=1`.

There is no lookahead anywhere in the chain, so a terminal chunk needs no flush:
any tail length is already exact.

### State memory per stream

With the config above, bf16:

* Transformer K/V: `8 layers x 2 x 16 heads x 71 frames x 64 head_dim x 2 B`
  = **2.22 MiB**
* Conv histories and transconv overlaps (pre_conv, 2x ConvNeXt, decoder chain at
  1024/1536/768/384/192/96 channels): **~0.26 MiB**

**~2.5 MiB per active stream**, ~80 MiB for 32 slots. This is the number T-PR8
reports at startup and per active stream.

## Baseline: what T-PR7 already provides

### #1756 — `incremental_codec.py`

* `Qwen3TTSIncrementalCodecState` — a dataclass holding `frame_position: int`,
  `transformer_context_length: int`, `transformer_keys` / `transformer_values`
  (`dict[int, Tensor]`, keyed by layer index), `conv_histories` and
  `transconv_overlaps` (`dict[str, Tensor]`, keyed by module path), plus
  `clone()`.
* `incremental_causal_conv1d` — rejects `stride != 1`, prepends the retained
  history, asserts the temporal length is preserved, retains `module.padding`
  steps.
* `incremental_causal_transconv1d` — runs the functional transposed conv with
  `bias=None`, overlap-adds the retained tail into the head of the output, emits
  `t * stride` samples, retains the new `right_pad` tail, and applies bias only
  to the emitted slice. Deferring bias this way is what keeps overlap-add from
  double-counting it; the formulation is exact and its buffer is fixed-size.
* `_incremental_transformer` — retains `window_size - 1 = 71` K/V frames per
  layer, which is the correct retention for a window that includes the query
  position.
* `Qwen3TTSIncrementalDecoder` — thorough `_require_attrs` structural validation
  of the loaded module tree (the decoder ships in the external `qwen-tts`
  package, pinned to `0.1.1`, so it is driven from outside rather than
  vendored), and `decode(codes, state)` restricted to `[1, Q, T]`.

### #1757 — opt-in serving path

* `enable_stateful_codec_decoder=False` on `create_vocoder_executor` and the
  scheduler.
* `_Qwen3TTSStreamState` gains `incremental_codec_state` and
  `incremental_codec_fallback`.
* `decode_delta` branches to `_decode_incremental_eager`, which clones the
  committed state, verifies `frame_position == ref_frames +
  emitted_generated_frames`, decodes only fresh frames, trims the reference
  prefix out of the first chunk's output, validates the exact delta length, and
  commits the candidate state only on success.
* `_prune_incremental_codes` keeps `stream_left_context_frames` of raw codes so
  a mid-request incremental failure can fall back to the left-context decoder
  without restarting the request or republishing PCM. **Preserve this.**
* Any exception sets `incremental_codec_fallback` for that request and logs.
* The flag forces `self._async_decode = False` and disables
  `_Qwen3TTSInitialDecodeGraphs`.

### What that leaves for T-PR8

Because #1757 forces `_async_decode = False`, the incremental path today runs
**synchronously on the caller thread, B=1, holding `_state_lock`**, with no
decode stream and no pinned staging. The entire async machinery —
`_initial_worker` / `_followup_worker`, `_initial_queue` /
`_followup_queue`, `_run_initial_batch` / `_run_followup_batch`,
`_group_decode_plans`, `_decode_group`, `_DecodeSlot`, `_Qwen3TTSDecodeHandle` —
is untouched by #1757. Reconnecting the incremental path to it is T-PR8's main
body of work and conflicts with neither draft.

`_Qwen3TTSDecodePlan` is reused as-is by #1757 (with
`window_start = consumed_frames`), so `_commit_decode_plan`'s bookkeeping —
playback deadlines, `next_decode_generated_frames`, `decoded_chunks` — keeps
working unchanged.

## T-PR8 prerequisites in #1756

Two changes are required inside the T-PR7 core. Both are small, both are in
someone else's PR, and both should be agreed on the #1756 review thread before
T-PR8 depends on them.

### P1 — per-row positions in the incremental Transformer

`_incremental_attention` currently derives its mask from one-dimensional,
batch-shared position vectors:

```python
allowed = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
scores = scores.masked_fill(~allowed.view(1, 1, *allowed.shape), ...)
```

`view(1, 1, q, k)` broadcasts across the batch, and `frame_position` /
`transformer_context_length` are scalar ints on the state. So B=1 is baked into
the mask, not merely guarded at the entry point: even with the
`codes.shape[0] != 1` check removed, every row of a batch would have to sit at
the identical absolute frame position with the identical context length. That
directly contradicts the roadmap requirement that streams at different playback
positions batch together.

Required change:

* `query_positions` / `key_positions` become `[B, q]` / `[B, k]`.
* `allowed` becomes `[B, q, k]`, applied as `allowed.view(B, 1, q, k)`.
* `frame_position` and `transformer_context_length` become per-row.

Everything else is already batch-generic: the conv and transposed-conv helpers,
`_repeat_kv`, and both matmuls. RoPE is shape-ready as well —
`rotary_emb(hidden_states, position_ids)` with `[B, q]` yields `cos`/`sin` of
`[B, q, dim]`, and `_apply_rotary_pos_emb`'s `unsqueeze(1)` broadcasts that
correctly against `[B, H, q, D]`. The change is roughly forty lines confined to
`_incremental_attention` and `_incremental_transformer`.

### P2 — declared state spec and explicit initialization

History buffers are materialized lazily on first use, taking dtype and device
from the incoming activation, so the key set is not knowable until after a
decode has run. An arena must preallocate. Required:

* `Qwen3TTSIncrementalDecoder.state_spec()` — enumerate every
  `conv_histories` / `transconv_overlaps` key with its channel count and history
  length, and the per-layer K/V shape, derived from the validated module tree.
* `init_state(batch_size, device, dtype)` — allocate a zeroed state from that
  spec.
* `state_bytes_per_stream()` — derived from the same spec, for the memory report.

## T-PR8 design

### `CodecStateArena`

Slot-indexed preallocated storage for the P2 state spec. Every buffer gains a
leading slot dimension `[S, ...]`, with `S` defaulting to the pipeline's
`max_running_requests` and independently configurable.

* K/V becomes `[S, L, H_kv, W-1, D]` plus per-slot `valid_len` and
  `frame_position`. Write at the tail and shift left on overflow; shifting 71
  frames is cheap and keeps the eager path readable. A ring buffer is deferred to
  T-PR9, where graph replay makes the shift a capture problem.
* **Slot lifecycle**: free list; acquire and zero on the bootstrap decode;
  release on finish or abort through `release_stream_resources`.
* **Release must not race the GPU.** A slot may only return to the free list
  after the completion event of the last decode that touched it has fired. Hang
  the slot reference off the existing `_Qwen3TTSDecodeHandle` keepalive set and
  free it inside `resolve()` and its failure paths. When completion cannot be
  proven, follow the existing `_CONTEXT_FATAL_RETAINED` / `slot.broken`
  precedent: the codec slot is retained for the life of the process, never
  recycled.
* **Exhaustion is not backpressure.** With no free slot, that request takes the
  left-context path and increments a counter. Admission limits are T-PR14's
  problem; this PR must not couple to them.

The arena does not attempt to make the decoder write in place. Cohorts are
executed by gather -> decode -> scatter, so the T-PR7 helpers may keep rebinding
tensors freely inside cohort scratch; the arena is the bounded, reportable,
reusable storage *between* steps. Eliminating the remaining per-step allocation
churn is a T-PR9 concern, because that is where it becomes a capture blocker
rather than an efficiency question.

### Cohort formation

**Steady state batches; bootstrap does not.** After the first decode,
`fresh_frames` equals the follow-up stride for every request, so steady-state
cohorts are naturally uniform. The bootstrap decode consumes
`ref_frames + initial_chunk_frames`, and `ref_frames` is request-dependent, so
bootstrap shapes are inherently ragged.

This maps onto the scheduler's existing split: the initial worker handles ragged
bootstrap decodes, the follow-up worker handles uniform steady-state decodes.
T-PR8 therefore keeps bootstrap at B=1 eager — matching #1757's behavior today —
and batches only follow-ups. Bucketing ragged reference lengths with padding and
masking is possible on top of P1 but is deliberately out of scope; it should be
measured against the bootstrap's share of total Codec time before it is built.

Cohort key is `fresh_frames` alone. A separate `COLD` / `WARM` mode dimension is
**not** needed at T-PR8: because arena K/V is always the full `W-1` window with
per-row `valid_len`, a cold stream and a warm stream have identical execution
shapes and differ only in their mask. A cold stream does some masked-out
attention work for its first ~71 frames; that is the price of uniform shapes and
it is bounded. Keep `mode` in the vocabulary for T-PR9's graph keys, where
skipping provably-masked K/V work may be worth a distinct capture.

Concretely, replace the current grouping

```python
groups.setdefault(tuple(entry[2].decoder_input.shape), []).append(entry)
```

with grouping on `fresh_frames`, and gather the cohort's slots into a
fixed-max-batch staging buffer. That staging buffer is deliberately the future
CUDA-graph replay buffer for T-PR9's `(mode, fresh_frames, batch_bucket)` keys:
T-PR8 builds the data path, T-PR9 captures it.

### Wiring into the async path

* Stop forcing `_async_decode = False` when the flag is on. Bootstrap decodes go
  to the initial worker, follow-ups to the follow-up worker, exactly as the
  left-context path does.
* Introduce `_IncrementalDecodePlan` (slot id, fresh codes, `fresh_frames`) and
  dispatch on it in `_group_decode_plans`, `_run_initial_batch`,
  `_run_followup_batch`, `_commit_initial`, and `_commit_followup`.
* **Restore deferred bad-row screening.** `_decode_incremental_eager` currently
  calls `_raise_for_bad_rows(bad_rows, 1)` inline, and `.nonzero()` on a device
  tensor forces a host sync. The async path exists precisely to defer that read
  into `resolve()` after the completion event; the batched incremental path must
  do the same, keeping the clamp-then-verify contract of
  `_screen_out_of_range_codes` intact.
* **Failure isolation within a cohort.** `_decode_group` already peels off rows
  named by `_Qwen3TTSInvalidCodeRows` and retries the remainder; that behavior
  carries over unchanged. For *other* exceptions, `_decode_group` currently
  fails every stream in the group. On the incremental path that is too harsh:
  the correct response is #1757's per-request `incremental_codec_fallback`
  applied to each cohort member, re-planned onto the left-context decoder, with
  the streams surviving. Slots for failed rows are released through the same
  event-gated path as normal completion.
* Preserve #1757's transactional commit: decode against gathered scratch and
  scatter back to the arena only after the delta length check passes.

### `_Qwen3TTSInitialDecodeGraphs`

#1757 disables it whenever the flag is on, because a bootstrap decode's input
shape is no longer `left_context + initial_chunk`. T-PR8 keeps it disabled on
the incremental path and records eager bootstrap as required T-PR9 coverage. The
flag is off by default, so the shipped path is unaffected.

### Stream-ordering invariant

Initial decodes run on `_decode_stream`, follow-ups on
`_followup_decode_stream`. Cross-stream access to a slot is safe today because
every commit happens after `resolve()`, which is a host synchronization point,
and the next decode for a request is only scheduled from inside that commit.
Write this invariant down as a comment plus an assertion: T-PR12's bounded async
decode is explicitly designed to relax host synchronization and would otherwise
break it silently.

### Observability

* Log `state_bytes_per_stream()` and the arena total at startup.
* Export active slots, per-stream state memory, left-context fallback count, and
  slot-exhaustion count.
* Keep the `codec_fresh_frames` / `codec_recomputed_context_frames` counters
  showing `recomputed == 0` on the incremental path — the roadmap's "show that
  the intended optimized path actually ran".

These are the first entries of the T-PR19 observability surface.

### Test matrix

Beyond the parity coverage #1756 and #1757 already carry:

* **Batched vs B=1 parity.** A cohort of streams at deliberately different frame
  positions and context lengths must produce, row by row, exactly what the same
  streams produce decoded individually. This is the test that proves P1.
* **Cold and warm in one cohort.** A stream on its first follow-up batched with
  a stream past 71 frames.
* Abort while a decode is in flight — the slot must not be recycled early.
* Every branch of `_handle_stream_done`, including a final with no fresh frames.
* Slot reuse: the state a reused slot starts from is provably zeroed.
* Out-of-order follow-up priority queue: gather picks the right slots.
* One bad row in a cohort fails only that row; one non-code exception in a
  cohort falls every member back to left-context without killing any stream.

### Reporting

* Performance against two baselines: the T-PR2 follow-up-graph path, and #1757's
  B=1 eager incremental path, at c1/c8/c16/c32.
* Accuracy unchanged from #1757 — T-PR8 adds no decoder math — but re-run the
  three-row comparison to prove batching did not perturb it.
* Additional persistent GPU memory: arena total and per-stream, measured, not
  computed.

## PR boundary

| | T-PR7 (#1756 + #1757, done) | T-PR8 (this spec) |
| :--- | :--- | :--- |
| decoder core | `incremental_codec.py`, B=1 | + P1 per-row positions, P2 state spec |
| state lives in | the request's `_Qwen3TTSStreamState` | slot-indexed `CodecStateArena` |
| execution | synchronous, B=1, under `_state_lock` | async workers, batched follow-ups |
| batching | none | cohorts keyed by `fresh_frames` |
| default | off | off (T-PR13 decides when it flips) |

Both keep `enable_stateful_codec_decoder` off by default and leave the
left-context path fully intact, so T-PR9 can capture graphs directly on the
T-PR8 staging buffers without touching the data path again.

## Non-goals

* CUDA graphs for the incremental path (T-PR9).
* Vocoder scheduling and deadline changes (T-PR10).
* Talker backpressure (T-PR11).
* Changing default chunk schedules or admission limits (T-PR6, T-PR14).
* Batching ragged bootstrap decodes across different reference lengths.
* Vendoring the `qwen-tts` decoder into this repository.
