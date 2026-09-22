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

The vocoder keeps an LRU cache of up to 32 speaker references, so switching back
to a cached reference reuses its conditioning. Inline references are keyed by
audio content; file references account for file metadata. Invalid references
fail instead of silently using the default.

Decoding runs in fixed windows of `chunk_tokens` codec tokens (25 by default,
one second of audio) plus three lookahead tokens. Flow attention and
convolution state carries across windows per request, and the reference's own
state is computed once and kept for up to four references. Chunks of the same
width from different requests, including different references and lengths, run
in one flow batch and one HiFT batch; HiFT re-vocodes the previous window's
last eight mel frames and cross-fades the overlap so window boundaries stay
seamless. Each row's flow state holds attention caches for every denoising step
over the reference plus its last 100 frames, so Code2Wav memory grows with
`max_batch_size` and reference length; lower `max_batch_size` or
`max_batch_cost` on small GPUs. Audio output remains non-streaming.
