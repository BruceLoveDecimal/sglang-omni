# MiniCPM-o Reference Audio

On the speech pipeline, pass an explicit speaker reference in
`audio.ref_audio` on `/v1/chat/completions`:

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:30000/v1", api_key="unused")
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
```

`stage_params.code2wav.ref_audio` is an alternative, with higher priority than
`audio.ref_audio`. Both accept `prompt_wav` as an alias. The Python pipeline
client can also supply `ref_audio` through `extra_params`. References must be
base64 audio data URIs, inline `{data, media_type}` descriptors, or encoded audio
bytes for the Python client. Paths and HTTP URLs are not fetched by this stage;
read or download the file on the client before sending it.

The reference conditions Token2wav's speaker embedding, prompt tokens, and mel
features. Audio supplied in chat messages remains understanding input and is not
automatically used as the speaker reference. Without an explicit reference,
Token2wav uses the checkpoint's `assets/HT_ref_audio.wav` when available.

The vocoder caches only the most recently used reference by audio content. A
different reference, including switching back to the default, rebuilds the
conditioning. Invalid references fail instead of silently using the default.
Audio output remains non-streaming.

## Chunked flow

By default the vocoder denoises each utterance in one flow pass with full
attention over the reference and the generated tokens. `chunked_flow` switches
the flow to the checkpoint's streaming formulation: the reference primes
per-timestep conformer and DiT caches, then every 25 tokens (plus 3 lookahead
tokens) are denoised against those caches, which are bounded to the reference
plus the most recent 100 frames. HiFT still vocodes the whole utterance once.

```bash
sgl-omni serve --model-path openbmb/MiniCPM-o-4_5 \
  --code2wav.factory.chunked_flow true
```

Chunked flow captures CUDA graphs for the DiT step at startup: one for the
regular chunk and one per 64-frame reference bucket up to 896 frames. Longer
references fall back to eager execution. `chunked_flow_cuda_graph=false` keeps
the chunked path eager for comparison. Chunked audio differs from the
whole-utterance output because later chunks are not visible to earlier frames;
compare quality before changing the default. Denoising 25-token chunks runs
many small DiT steps, so on a single request it is slower than the
whole-utterance pass; the option exists for bounded caches and as the basis for
streaming audio output.
