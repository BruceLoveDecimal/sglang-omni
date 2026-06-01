# Ming-Omni GSM8K Text Streaming Benchmark

Run date: 2026-06-01
Remote host alias: `sglang-omni-h20`
Dataset: GSM8K test split, first 32 measured examples after 1 warmup

## Request Shape

Each measured request used OpenAI `/v1/chat/completions` SSE streaming with:

```json
{
  "model": "ming-omni",
  "stream": true,
  "modalities": ["text"],
  "temperature": 0,
  "max_tokens": 256
}
```

## Server Configuration

- GPU: 2 x NVIDIA H20, 95 GiB each
- `tp-size`: 2
- `cpu-offload-gb`: 60
- `mem-fraction-static`: 0.80
- Relay backend: `shm`
- CUDA visible devices: `0,1`

## Summary

- Successes: 32/32
- Failures: 0/32
- Every measured request emitted multiple non-empty `delta.content` chunks.
- Non-empty content chunks:
  - mean: 236.0
  - median: 256.0
  - min/max: 165 / 256
- TTFT:
  - mean: 5.733 s
  - median: 5.455 s
  - p95: 7.656 s
  - min/max: 4.241 s / 8.378 s
- TPOT:
  - mean: 1.824 s/token
  - median: 1.826 s/token
  - p95: 1.833 s/token
  - min/max: 1.794 s/token / 1.846 s/token
- Latency:
  - mean: 435.988 s
  - median: 471.070 s
  - p95: 474.022 s
  - min/max: 308.022 s / 475.016 s
- Completion tokens:
  - mean: 236.406
  - median: 256
  - min/max: 166 / 256

## Artifacts

- `summary.json`: aggregate metrics
- `metadata.json`: benchmark configuration
- `requests.jsonl`: per-request metrics, prompts, expected answers, and generated text
- `benchmark.txt`: runner stdout
- `server.txt`: server log
- `benchmark_ming_gsm8k_stream.py`: benchmark script
- `run_ming_gsm8k_stream_h20.sh`: remote runner

## Notes

- This is a 32-example GSM8K streaming sweep, not the full 1319-example test split.
- The run is intended to validate issue #600's streaming behavior with explicit `modalities: ["text"]`: TTFT and TPOT are now measurable from incremental SSE chunks.
- The same H20 MoE kernel fallback warning appears in the server log, so absolute throughput numbers should be interpreted with that fallback and CPU offload configuration in mind.
