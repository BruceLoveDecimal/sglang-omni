# Ming-Omni Text Streaming E2E Results

Remote run directory:

```text
/root/autodl-tmp/sglang-omni-h20/results/ming_text_stream_smoke_bench_20260601_185728
```

Environment:

- Hardware: 2x NVIDIA H20, GPU0-GPU1 topology `NV18`
- Code branch: `feat/ming-omni-text-streaming-600`
- Code commit: `120e601242c7a5374a0d9c2b3c3ed19517ea1143`
- Model path used at runtime: `/root/ming-model`
- Dataset: local GSM8K JSONL at `/root/autodl-tmp/sglang-omni-h20/data/gsm8k_test.jsonl`

Smoke result:

- `limit=1`, `concurrency=1`, `max_tokens=64`
- `successes=1`, `failures=0`
- `streaming_successes=1`, `streaming_failures=0`
- TTFT mean: `2.989696517586708s`
- TPOT mean: `1.7983033621003703s`

Benchmark result:

- `limit=32`, `warmup=1`, `concurrency=4`, `max_tokens=256`
- `successes=32`, `failures=0`
- `streaming_successes=32`, `streaming_failures=0`
- exact match: `17/32` (`0.53125`)
- TTFT mean/median/p95: `4.897120159119368s` / `5.425483651459217s` / `5.881298181042075s`
- TPOT mean/median/p95: `1.907550052852541s` / `1.8865111236697902s` / `2.025891140262111s`
- Latency mean/median/p95: `456.0590956188971s` / `469.8824125416577s` / `514.5407590597868s`

Note: after the run, SSH started closing connections before key exchange, so the
full remote `requests.jsonl` and `server.log` were not copied into this local
artifact. The summaries and logs here are reconstructed from the completed runner
stdout captured during the run.
