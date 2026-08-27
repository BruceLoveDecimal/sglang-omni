# Qwen3-TTS stateful incremental Codec decoding (T-PR7 / T-PR8)

Spec for two stacked PRs from the Qwen3-TTS optimization roadmap
([#1754](https://github.com/sgl-project/sglang-omni/issues/1754)):

* **T-PR7** — a stateful incremental Codec decoder, B=1 eager, correctness first.
* **T-PR8** — integrate it into `Qwen3TTSStreamingVocoderScheduler` with
  slot-indexed batched state management.

They land as two PRs. T-PR8 depends on T-PR7 and adds no new decoder math; it
only changes *where the state lives* and *how many streams execute per launch*.

## Problem

Every streaming decode today rebuilds a left-context window and reruns the whole
speech-tokenizer decoder over it. `_build_decode_plan` in
`sglang_omni/models/qwen3_tts/streaming_vocoder.py` slices

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
ever sees 24 frames. So today's streaming output is **already an approximation**
of full-sequence decoding: the Transformer context is truncated to a third of
the window the model was trained with. Stateful decoding removes that truncation
as a side effect, so the incremental path is expected to be *closer* to the
full-sequence reference than the current streaming path, not merely equal to it.
This must be stated explicitly in the accuracy report, and the comparison must
carry three rows: full-sequence, current streaming, incremental streaming.

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
| `pre_transformer` | sliding-window attn, 8 layers | per-layer K/V + absolute frame position | <=72 frames |
| `upsample[i][0]` | ConvTranspose1d k=s=`ratio` | **none** | — |
| `upsample[i][1]` | ConvNeXt, dwconv k=7 | input history | 6 steps |
| `decoder[0]` | CausalConv1d k=7 | input history | 6 steps |
| `DecoderBlock[i]` head | ConvTranspose1d k=2r s=r | input history | 1 step |
| `DecoderBlock[i]` residuals | CausalConv1d k=7, dilation 1/3/9 | input history | 6 / 18 / 54 steps |
| `decoder[-1]` | CausalConv1d k=7 | input history | 6 steps |
| `SnakeBeta`, `LayerNorm`, `RMSNorm`, MLPs | elementwise / per-position | none | — |

Four structural properties make a *bit-faithful* incremental decoder possible,
rather than an approximation that needs a quality argument:

1. **Zero-initialized state is an exact cold start.** Every causal conv in this
   decoder left-pads with `F.pad(..., value=0)`. Initializing every history
   buffer to zeros is therefore identical to what the full-sequence path does at
   `t = 0`. The COLD path needs no special case and no warm-up discard.

2. **ConvTranspose overlap spans exactly one input step.** In `DecoderBlock` the
   transposed conv has `kernel_size = 2r`, `stride = r`, and the module trims
   `right_pad = kernel_size - stride = r` from the tail. Output block
   `[j*r, (j+1)*r)` therefore depends only on input steps `j` and `j-1`.
   Feeding `1 + t` steps (one retained history step plus `t` fresh) yields
   `(t+1)*r` samples after the module's own right trim; discarding the leading
   `r` samples leaves exactly the `t*r` samples the full-sequence run would have
   produced. The tail that the module trims is the part that gets recomputed on
   the next call with the same history step, so nothing is lost and nothing is
   double counted. The `upsample` stage transposed convs have `k == s`, produce
   no overlap at all, and are stateless.

   Retaining one input step is preferred over an explicit overlap-add buffer:
   the execution shape stays a pure function of `fresh_frames`, which is what
   T-PR9 needs in order to key CUDA graphs by `(mode, fresh_frames, batch)`.

3. **RoPE uses absolute positions.** Keys are rotated by absolute position
   before entering the cache, so evicting keys outside the sliding window is
   exact. The state must carry `frames_seen` and pass absolute
   `cache_position` / `position_ids` on every step — this is the roadmap's
   "frame position / valid context metadata". The per-step mask must reproduce
   `create_sliding_window_causal_mask` semantics for the same absolute
   positions; a partition that crosses the 72-frame boundary is the test that
   proves it.

4. **`extra_padding` is always zero here.** For `stride == 1`,
   `_get_extra_padding_for_conv1d` reduces to `0` identically. Every
   `Qwen3TTSTokenizerV2CausalConvNet` in the decode path uses the default
   `stride=1`. Assert this at construction so a future checkpoint that breaks
   the assumption fails loudly instead of silently shifting samples.

There is no lookahead anywhere in the chain, so a terminal chunk needs no flush:
any tail length is already exact. The roadmap's "terminal chunk" validation is
therefore a test that this claim holds, not a code path.

### State memory per stream

With the config above, bf16:

* Transformer K/V: `8 layers x 2 x 16 heads x 72 frames x 64 head_dim x 2 B`
  = **2.25 MiB**
* Conv histories (pre_conv, 2x ConvNeXt, decoder chain at 1024/1536/768/384/192/96
  channels): **~0.26 MiB**

**~2.5 MiB per active stream**, ~80 MiB for 32 slots. This is the number T-PR8
reports at startup and per active stream.

## T-PR7 — incremental decoder core

### Placement

The decoder ships in the external `qwen-tts` package (pinned to `0.1.1` by
`sglang_omni/models/qwen3_tts/stages.py`). Vendoring `modeling_qwen3_tts_tokenizer_v2.py`
into this repo would fork a file we do not own and would have to be re-synced on
every upstream bump. Instead, drive the loaded modules from outside:

`sglang_omni/models/qwen3_tts/incremental_codec.py`

```python
@dataclass
class CodecDecoderState:
    """Per-stream causal state. All buffers fixed-size; none grow with chunk."""
    frames_seen: int
    kv: ...                 # per-layer K/V, <= sliding_window frames
    conv_history: dict      # keyed by module path
    transconv_history: dict # 1 input step per DecoderBlock head


class StatefulCodecDecoder:
    def __init__(self, decoder): ...
        # Structural validation: walk the loaded module tree and check every
        # module type, kernel_size, stride, dilation, groups and padding against
        # what this implementation assumes. Any mismatch raises; the caller
        # falls back to the left-context path.

    def init_state(self, batch_size, device, dtype) -> CodecDecoderState: ...
    def decode_incremental(self, codes, state) -> torch.Tensor:
        """codes [B, Q, t_fresh] -> wav [B, 1, t_fresh * total_upsample]"""
    def state_bytes_per_stream(self) -> int: ...
```

### Implementation notes

* **Causal conv**: replace `F.pad(x, (padding, 0))` with
  `conv(cat([history, x]))`, then `history = cat([history, x])[..., -padding:]`.
  The `cat` form handles `t_fresh < padding` naturally, so 1-frame chunks need
  no special case.
* **Transposed conv**: prepend the retained input step, run the module (which
  applies its own right trim), drop the leading `r` samples when the history is
  non-empty. Bias is applied by the module as usual; because we drop whole
  output blocks rather than summing partial ones, there is no double-counted
  bias — this is the failure mode an overlap-add formulation has to work around.
* **Transformer**: at T-PR7, a HF `DynamicCache` trimmed to `sliding_window`
  after each step, with explicit absolute `cache_position`. Pre-allocated K/V is
  a T-PR8 concern; keeping it simple here keeps the parity tests honest.
* **Reference prefix (ICL)**: no separate prefix cache. Reference frames are
  simply the first input fed into a fresh state; their output samples are
  discarded. This matches the existing first-window behavior over
  `[0, ref_frames + initial_chunk)`.
* **B=1 only**: `decode_incremental` rejects `B > 1`. Batching is T-PR8.

### Wiring (kept minimal in this PR)

Add `enable_incremental_codec: bool = False` to
`Qwen3TTSStreamingVocoderScheduler.__init__` and to `create_vocoder_executor`,
reachable through `FactoryArgs`. When enabled:

* `_build_decode_plan` emits an incremental plan (fresh frames + the request's
  `CodecDecoderState`) instead of a left-context window; `_extract_delta`
  becomes the identity because every returned sample is new.
* The existing `code_chunks` retention and pruning stay **unchanged**, so a
  mid-stream incremental failure can fall back to the left-context path and
  continue the same request without restarting it or republishing PCM.
* Commit the new state only after the decode returns an output of the exact
  expected length. Decode against a cloned state and swap on success, so a
  failed step cannot leave a half-advanced stream.
* Deterministic mode is unaffected: incremental decode is already B=1, matching
  the per-request isolation #1475 requires.

Default off. Nothing about the current path changes when the flag is not set.

### Validation

`tests/unit_test/qwen3_tts/test_incremental_codec.py`:

* **Parity under arbitrary partitions.** Build a small decoder with the real
  module classes when `qwen_tts` is importable (skip otherwise), fp64 on CPU,
  and assert `decode_full(codes) == cat(decode_incremental(chunk) for chunk)`
  bit-for-bit across seeded random partitions. The partition set must include:
  1-frame chunks; chunks longer than `sliding_window`; a partition crossing the
  72-frame boundary; an odd terminal chunk; a reference prefix; cold start.
* **Partition invariance of state.** After N frames, the state is independent of
  how those frames were partitioned.
* **Bounded state.** Buffer sizes are constant after the first steps, and
  `state_bytes_per_stream()` matches measured allocation.
* **Structural validation.** A mutated module tree (wrong kernel, wrong stride,
  a `stride != 1` causal conv) raises rather than producing wrong audio.
* bf16 / GPU parity with a stated tolerance, reported separately from the fp64
  exactness result.

### Reporting

* Accuracy: WER and speaker similarity for full-sequence, current streaming, and
  incremental streaming. Explain the expected direction of the incremental /
  current-streaming difference (see Problem).
* Performance vs the T-PR2 follow-up-graph baseline at c1/c8/c16/c32.
  **Eager incremental decoding may not beat an already-graphed left-context
  decode at small chunk sizes.** That does not block T-PR7, which is a
  correctness PR; graph coverage is T-PR9. Report it honestly rather than
  choosing a chunk size that flatters the new path.
* Add `codec_fresh_frames` and `codec_recomputed_context_frames` counters and
  show `recomputed == 0` on the incremental path — this is the roadmap's
  "show that the intended optimized path actually ran".

## T-PR8 — vocoder integration and batched state

### `CodecStateArena`

Replace the per-request Python tensor dict from T-PR7 with slot-indexed
pre-allocated storage. Every buffer from the T-PR7 state spec gains a leading
slot dimension `[S, ...]`, with `S` defaulting to the pipeline's
`max_running_requests` and independently configurable.

* Transformer K/V becomes `[S, L, H_kv, W, D]` plus per-slot `valid_len` and
  `frames_seen`. At T-PR8 use a linear layout — write at the tail, shift left on
  overflow. Shifting 72 frames is cheap and keeps the eager path readable; a
  ring buffer is a T-PR9 change once graph replay demands it.
* **Slot lifecycle**: free list; acquire and zero on the first (COLD) decode;
  release on finish or abort through `release_stream_resources`.
* **Release must not race the GPU.** A slot may only return to the free list
  after the completion event of the last decode that touched it has fired. Hang
  the slot reference off the existing `_Qwen3TTSDecodeHandle` keepalive set and
  free it inside `resolve()` (and its failure paths). When completion cannot be
  proven, follow the existing `_CONTEXT_FATAL_RETAINED` / `slot.broken`
  precedent: the codec slot is retained for the life of the process, never
  recycled.
* **Exhaustion is not backpressure.** With no free slot, that request falls back
  to the left-context path and increments a counter. Admission limits are
  T-PR14's problem; this PR must not couple to them.

### Batched execution

* Group plans by `(mode, fresh_frames)` where `mode` is `COLD` or `WARM`, rather
  than by `decoder_input.shape` as `_group_decode_plans` does today.
* **Streams at different playback positions batch together.** State buffers are
  fixed-size and position differences are expressed as per-row attention masks
  and per-row `frames_seen`, not as different tensor shapes. Conv histories are
  homogeneous because of the zero-initialization property, so they need no mask
  at all. This is what makes the roadmap's "different playback/frame positions
  batch together" true rather than aspirational.
* Execution is gather -> decode -> scatter: gather the cohort's slots into a
  fixed-max-batch staging buffer, run, scatter back. That staging buffer is
  deliberately the future CUDA-graph replay buffer for T-PR9's
  `(mode, fresh_frames, batch_bucket)` keys; T-PR8 builds the data path, T-PR9
  captures it.
* Deterministic mode keeps B=1 per request, consistent with the existing policy.

### Stream-ordering invariant

Initial decodes run on `_decode_stream`, follow-ups on
`_followup_decode_stream`. Cross-stream state access is currently safe because
every commit happens after `resolve()`, which is a host synchronization point,
and the next decode for a request is only scheduled from inside that commit.
Write this invariant down as a comment plus an assertion, because T-PR12's
bounded async decode is explicitly designed to relax host synchronization and
would otherwise break it silently.

### Scheduler changes

All inside `Qwen3TTSStreamingVocoderScheduler`:

* New `_IncrementalDecodePlan` (slot id, fresh codes, mode); dispatch in
  `_group_decode_plans`, `_run_initial_batch`, `_run_followup_batch`,
  `_commit_initial`, `_commit_followup`.
* `_Qwen3TTSInvalidCodeRows` semantics are unchanged; a failed row releases its
  slot through the abort path and must not disturb the other rows in its cohort.
* `_Qwen3TTSInitialDecodeGraphs` stops matching once COLD decodes go
  incremental, because the input shape changes. Accept eager COLD while the flag
  is on and record it as required T-PR9 coverage. The flag is off by default, so
  this is not a regression of the shipped path.

### Observability

* Log `state_bytes_per_stream()` and the arena total at startup.
* Export active slots, per-stream state memory, left-context fallback count, and
  slot-exhaustion count. These are the first entries of the T-PR19
  observability surface.

### Lifecycle test matrix

* Abort while a decode is in flight (slot must not be recycled early).
* Every branch of `_handle_stream_done`, including a final with no fresh frames.
* Slot reuse: the state a reused slot starts from is provably zeroed.
* Out-of-order follow-up priority queue: gather picks the right slots.
* One bad row in a cohort fails only that row.

## PR boundary

| | T-PR7 | T-PR8 |
| :--- | :--- | :--- |
| adds | `incremental_codec.py`, parity tests, B=1 flag wiring | `CodecStateArena`, batched gather/scatter, lifecycle, counters |
| state lives in | the request's `_Qwen3TTSStreamState` | slot-indexed arena |
| batching | none (B=1 eager) | cohorts keyed by `(mode, fresh_frames)` |
| default | off | off (T-PR13 decides when it flips) |
| measured against | T-PR2 baseline, full-sequence accuracy, partition invariance | T-PR7 B=1, c8/c16/c32 throughput, cancellation and reuse stress |

Both PRs keep `enable_incremental_codec` off by default and leave the
left-context path fully intact, so T-PR9 can capture graphs directly on the
T-PR8 staging buffers without touching the data path again.

## Non-goals

* CUDA graphs for the incremental path (T-PR9).
* Vocoder scheduling and deadline changes (T-PR10).
* Talker backpressure (T-PR11).
* Changing default chunk schedules or admission limits (T-PR6, T-PR14).
* Vendoring the `qwen-tts` decoder into this repository.
