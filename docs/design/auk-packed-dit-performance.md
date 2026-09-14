# AuK packed DiT with compiled blocks and CUDA graphs

This experiment adds opt-in packed execution to the version of
[#2122](https://github.com/sgl-project/sglang-omni/pull/2122) at
`2dc090b75e6b53d25f1707c781c0a24b9389cb31`.

Both comparison arms enable `enable_dit_torch_compile` and
`enable_dit_cuda_graph`; only `enable_packed_dit` changes. The packed option
remains disabled by default.

The transformer trunk operates on valid tokens, including projections, norms,
FFNs and residuals. Input embeddings and the FP32 Euler state remain padded.
A custom operator keeps FlashAttention dispatch outside Dynamo tracing.
Packed layout tensors are graph inputs so replay refreshes indices and
`cu_seqlens` for the current requests.

## Environment and results

Measured on 2026-09-14 with one RTX 5090 (32 GB), driver 580.105.08,
PyTorch 2.13.0+cu130 and SGLang 0.5.19. The real AuK base model uses a BF16
backbone, CFG=2 and NFE=32.

Three serving pairs ran in A/B, B/A, A/B order with a fresh server per arm.
Each accepted window contains eight requests at c1/c2 or 32 requests at
c8/c16, using the same SeedTTS English voice-cloning inputs and seed 1234.
The percentages below are means of paired ratios.

| Concurrency | #2122 QPS, three runs | + packed QPS, three runs | QPS gain | p99 change |
|---|---|---|---:|---:|
| c1 | 1.231 / 1.243 / 1.262 | 1.268 / 1.251 / 1.253 | +0.98% | -1.02% |
| c2 | 1.732 / 1.733 / 1.737 | 1.738 / 1.739 / 1.739 | +0.27% | -0.94% |
| c8 | 1.713 / 1.700 / 1.698 | 2.304 / 2.284 / 2.225 | +33.30% | -32.18% |
| c16 | 1.602 / 1.597 / 1.577 | 2.396 / 2.362 / 2.366 | +49.17% | -37.20% |

All 480 timed requests succeeded. Accepted windows had no new graph captures
or graph warnings. Warmup initially used eight c1/c2 requests and 32 c8/c16
requests, followed by retries if new shapes were captured. One incomplete
service group was discarded after six unsuccessful attempts to obtain a
capture-free c8 window. The remaining three services expanded the c1 warmup
to all 32 samples. Dynamic grouping, differing warmup histories and selection
of capture-free windows limit this small experiment; these are warm results,
not cold-start or production p99 estimates.

The original graph policy still captures only actual batches up to two.
Actual batches were all singleton at c1/c2, and at most seven/eight at c8/c16.
No accepted serving window contained batch=2, so the large-concurrency gain
primarily measures packed + compile alongside singleton graph replay.

A separate fixed-shape, real-weight batch=2 test did exercise packed + graph:
32-step sampling decreased from 0.7987 to 0.5354 seconds (1.492x), using the
median of three pairs after warmup. Mixed-length batch=8/16 sampling improved
by 1.756x/1.779x; equal-length batches improved by 1.103x/1.095x.
These larger batches were not graph-captured.

## Validation and limits

- All 53 AuK unit tests passed, including packed/padded semantic comparisons
  and actual FlashAttention checks.
- A compiled GPU fixture reused one packed graph across the original layout,
  reversed request order and changed masks. Four-step outputs matched eager
  packed execution exactly in all three cases.
- All 480 audio outputs were finite, nonempty, 24 kHz and matched their paired
  output lengths. Qwen3-ASR-1.7B, using the official Transformers SDPA backend
  and repository WER normalization, scored both arms at 0.3597% corpus WER
  (9 errors / 2502 words). Two paired errors changed in opposite directions;
  aggregate equality does not establish waveform or perceptual equivalence.
  There were only 32 independent texts and no listening or speaker-similarity
  evaluation.
- This does not implement general total-token bucket graphs. Packed tensor
  shapes and scalar layout metadata participate in the graph key, retaining
  the original 32-graph cache limit and memory checks.
- H100/H200 and c32/c64 were not tested. First-use compilation and capture
  costs are excluded from the warm timing results.
