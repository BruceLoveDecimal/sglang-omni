# Vendored MOSS Audio Tokenizer (with streaming KV-cache decode)

This directory vendors the MOSS-TTS Stage-1 codec so that sglang-omni owns a
**stateful, KV-cache streaming** decode path. See
`docs/design/moss_tts_stateful_codec_streaming_rfc.en.md` for the rationale.

## Provenance / 来源

| | |
|---|---|
| **Upstream model** | `OpenMOSS-Team/MOSS-Audio-Tokenizer` (HuggingFace, `trust_remote_code`) |
| **Adapted from** | `vllm-omni` `vllm_omni/model_executor/models/moss_tts/audio_tokenizer.py` |
| **vllm-omni commit** | `b550709b` (file), repo HEAD `6ddaf6188d1455ce306e88113610eb53dedb9473` |
| **Upstream license** | Apache-2.0 |
| **Vendored on** | 2026-06-09 |

## Local modifications / 本地改动

The upstream/vllm-omni copy is **inference-only, single-pass batch decode**; its
header notes that "Streaming KV-cache infrastructure removed". This vendored copy
**re-introduces streaming**:

- `_apply_rope(..., pos_offset=0)` — RoPE positions continue across chunks.
- `_Attention.forward(x, kv=None)` — per-layer KV cache; returns updated cache.
- `_TransformerLayer` / `_Transformer` / `_ProjectedTransformer` thread a
  per-layer cache list.
- `StageCache` / `StreamingCache` dataclasses — one `StageCache` per decoder
  transformer stage, each with its own `pos_offset` (stages run at different time
  rates because patch upsampling changes T).
- `MossAudioTokenizerModel.decode_stream(codes_chunk, state, last_chunk)` —
  stateful streaming decode, O(N) total, output equal to `batch_decode` per
  segment within numerical tolerance.

`batch_encode` / `batch_decode` (the non-streaming path) are preserved unchanged
in behavior.

## Version pinning / 版本锁定

The codec checkpoint and this modeling code must stay in lockstep. Pin the
checkpoint revision wherever it is loaded (`MOSS_TTS_CODEC_PATH` / config) and do
not bump the checkpoint without re-verifying the parity tests in
`tests/unit_test/moss_tts/`.
