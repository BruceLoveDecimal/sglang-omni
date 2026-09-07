# Nemotron 3.5 ASR

[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is a multilingual RNN-T speech-recognition model served through the
OpenAI-compatible `/v1/audio/transcriptions` endpoint. SGLang-Omni resamples
each uploaded file to mono 16 kHz audio and batches compatible requests into
one model `generate()` call.

This integration accepts complete uploaded audio files. Client-driven
incremental PCM ingestion and persistent cross-chunk model caches are outside
the scope of this offline integration.

Nemotron 3.5 ASR does not support `/v1/audio/translations`; use
`/v1/audio/transcriptions`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download the model:

```bash
hf download nvidia/nemotron-3.5-asr-streaming-0.6b
```

## Server Configuration

Nemotron 3.5 ASR runs as one model-owned ASR stage on one GPU. The validated
default dtype is `float32`. The scheduler admits up to eight compatible
requests to one model batch and waits for at most 2 ms to form that batch.

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --port 8000
```

The checkpoint supports lookahead values `0`, `3`, `6`, and `13`; the default
is `3`. Configure lookahead and batching on the ASR stage when needed:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 \
  --asr.factory.max_batch_wait_ms 2 \
  --port 8000
```

Requests with different explicit `max_new_tokens` values are placed in
separate model batches so each request keeps its requested output limit.

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F file=@tests/data/query_to_cars.wav \
  -F language=auto \
  -F response_format=verbose_json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "language": "auto",
            "response_format": "verbose_json",
        },
        files={"file": ("query_to_cars.wav", audio_file, "audio/wav")},
        timeout=300,
    )

response.raise_for_status()
print(response.json())
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `auto` | Checkpoint-defined locale or language code, matched case-insensitively; `auto` enables language detection |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0` | Nemotron uses greedy RNN-T decoding and rejects non-zero values |
| `max_new_tokens` | integer | model default | Optional positive RNN-T step limit (including blank emissions) |
| `prompt` | string | unset | Text prompts are not supported; non-empty values are rejected |

With `language=auto`, `verbose_json.language` is populated when the model emits
one unambiguous locale tag. Locale tags are removed from the returned clean
transcript. If the output contains multiple different locale tags, the API
does not invent a single language for the response.

## Batching and Limitations

- Audio is prepared through the shared mono 16 kHz transcription path.
- The processor's checkpoint-provided prompt dictionary is authoritative;
  unsupported language values fail before model inference.
- The model's generation path owns mutable encoder and decoder state, so model
  calls are serialized while each admitted scheduler batch is executed as one
  true batched `generate()` call.
- This integration does not expose live incremental audio ingestion. Upload the
  complete audio file in each transcription request.

## Apple Silicon (MLX)

Run `./install.sh` and activate `.venv-apple`. Set `SGLANG_USE_MLX=1` to
select the native MLX FastConformer, language projector, LSTM predictor and
RNN-T joint network. The shared processor prepares log-mel features on CPU;
`mlx-audio` and NeMo are not runtime dependencies.

The MLX profile uses FP32, greedy decoding, one active request, and at most
60 seconds per uploaded file. The scheduler clamps `max_batch_size` to one
while preserving queued requests and their independent decoder state.
`json`, `text`, `verbose_json`, and SSE completion responses are supported.
SSE emits the final transcript after offline inference; this path does not
provide incremental audio ingestion or cache-aware live transcription.

Official Hugging Face safetensors load directly, without a separate MLX
conversion or quantization step:

```bash
source .venv-apple/bin/activate
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/ffmpeg@7/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.dtype float32 \
  --port 8000
```

### Download weights from ModelScope

The ModelScope mirror supplies the original `.nemo` archive. Convert its
learned parameters to the official safetensors layout once. The converter
uses Torch's weights-only loader, verifies every parameter name and shape,
and writes a source SHA-256 manifest. It obtains only tokenizer and processor
metadata from the pinned official Hugging Face revision, not model weights.
Use `--metadata-path` to supply that metadata locally.

```bash
mkdir -p "$HOME/models/nemotron-nemo"
curl -fL --retry 5 -C - \
  'https://modelscope.cn/models/nv-community/nemotron-3.5-asr-streaming-0.6b/resolve/master/nemotron-3.5-asr-streaming-0.6b.nemo' \
  -o "$HOME/models/nemotron-nemo/nemotron-3.5-asr-streaming-0.6b.nemo"

python -m sglang_omni.models.nemotron3_5_asr.convert_nemo \
  --nemo-path "$HOME/models/nemotron-nemo/nemotron-3.5-asr-streaming-0.6b.nemo" \
  --output "$HOME/models/nemotron-3.5-asr-0.6b"

SGLANG_USE_MLX=1 sgl-omni serve \
  --model-path "$HOME/models/nemotron-3.5-asr-0.6b" --port 8000
```

The same converted checkpoint can be loaded by the Torch runner for numerical
comparison. Setting `SGLANG_USE_MLX=0` preserves the existing Torch path;
Torch/MPS is not qualified by the MLX validation below.

### Reproduce validation

```bash
SGLANG_USE_MLX=1 python -m pytest tests/unit_test/nemotron3_5_asr -q
python -m benchmarks.eval.verify_nemotron_mlx \
  --model-path "$HOME/models/nemotron-3.5-asr-0.6b" \
  --output /tmp/nemotron-validation.json
```

The e2e verifier checks complete token sequences against the Torch CPU
reference for both bundled speech clips, explicit English and automatic
language detection, and lookahead `0`, `3`, `6`, and `13`. It then launches
an actual MLX server, verifies HTTP response formats and SSE completion,
queued/repeated requests, invalid inputs and recovery, and stops the server.
The JSON report records package versions, revision, transcripts, timings,
and MLX peak memory. These short-clip checks do not establish corpus-level
WER or sustained-load performance.

#### Local validation snapshot (2026-09-07)

Apple M1 Pro, 32 GiB unified memory; FP32; MLX 0.32.2, Torch 2.11.0,
Transformers 5.12.1, SGLang source `71de97b264b04dcd514cf904003028aefe9775c8`
(installer's v0.5.18 source). The ModelScope archive SHA-256 was
`210214ed94039bf6bfbb9a047c7fa289628db75b103e2bf6381fa78285436a74`;
all 655 learned parameter tensors passed conversion name/shape checks.

| Check | Result |
|---|---|
| Complete Torch CPU / MLX token sequences | 16/16 exact matches |
| HTTP/SSE requests | 25/25 expected responses: 19 HTTP 200, 6 HTTP 400 |
| Locale cleanup and `verbose_json.language` | Explicit English and automatic detection passed |
| Concurrent/repeated requests and recovery after invalid input | Passed |
| Warm serial HTTP observations, 4.620 s / 5.064 s speech clips | Approximately 0.120–0.164 s total request time |
| MLX peak allocation during component validation | 2.62 GB (excludes Torch CPU and HTTP processes) |

Timings are smoke-test observations across response formats, not a throughput
benchmark. The reproducible JSON report includes source-file and audio hashes
so results can be tied to the implementation even when run from a dirty tree.
