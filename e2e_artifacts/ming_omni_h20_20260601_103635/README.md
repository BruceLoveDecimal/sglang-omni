# Ming-Omni Dual H20 E2E Results

Run date: 2026-06-01
Remote host alias: `sglang-omni-h20`
Model path: `/root/autodl-tmp/sglang-omni-h20/models/Ming-flash-omni-2.0`

## Server Configuration

- GPU: 2 x NVIDIA H20, 95 GiB each
- Command: `examples/run_ming_omni_server.py`
- `tp-size`: 2
- `cpu-offload-gb`: 60
- `mem-fraction-static`: 0.80
- Relay backend: `shm`
- CUDA visible devices: `0,1`

## Results

- Text streaming e2e: passed
  - SSE chunks with content: 42
  - Finish reason: `stop`
  - Usage: `prompt_tokens=31`, `completion_tokens=43`, `total_tokens=74`
- ASR functional e2e: passed, 4/4 prompts
  - `transcribe`: 102.886 s
  - `transcribe_zh`: 18.495 s
  - `understand`: 104.359 s
  - `summarize`: 16.199 s
- ASR benchmark: passed, 5/5 measured runs
  - Latencies: 98.993 s, 98.998 s, 98.891 s, 98.994 s, 98.927 s
  - Mean: 98.961 s
  - Median: 98.993 s
  - P95: 98.997 s
  - Std: 0.044 s

## Artifacts

- `ming_text_stream_20260601_105615.sse`: raw streaming response
- `ming_asr_e2e_20260601_105752.json`: ASR functional results
- `ming_asr_benchmark_20260601_110154.json`: ASR benchmark results
- `ming_server_20260601_103635.txt`: server log

## Notes

- Cold startup took roughly 20 minutes on this instance before `/v1/models` became ready, dominated by loading/offload initialization.
- The server log includes SGLang warnings that no NVIDIA H20-specific Triton 3.5.1 MoE kernel config was found, so it fell back to the Triton 3.4.0 config. Benchmark numbers should be interpreted with that fallback in mind.
- The server was stopped after the run and both GPUs were confirmed idle.
