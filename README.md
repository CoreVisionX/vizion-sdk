# Vizion SDK

Python SDK for [Vizion.fast](https://www.vizion.fast) — real-time GPU segmentation over WebSocket.

## Install

```bash
pip install git+https://github.com/CoreVisionX/vizion-sdk.git
```

With OpenCV + numpy for mask decoding and webcam demos:

```bash
pip install "vizion[cv] @ git+https://github.com/CoreVisionX/vizion-sdk.git"
```

Requires Python 3.10+.

## Quick Start

```python
import os
from vizion import VizionClient

client = VizionClient(os.environ["VIZION_API_KEY"])
client.connect()

with open("frame.jpg", "rb") as f:
    jpeg_bytes = f.read()

result = client.segment(jpeg_bytes, prompts=["person", "car"])

for det in result.results:
    print(f"{det.prompt}: {len(det.instances)} found")
    for inst in det.instances:
        print(f"  bbox=({inst.x1},{inst.y1})-({inst.x2},{inst.y2})  conf={inst.confidence:.2f}")

print(f"Latency: {result.decode_segment_ms:.1f}ms")

client.close()
```

Use a context manager to ensure the session always shuts down:

```python
with VizionClient(os.environ["VIZION_API_KEY"]) as client:
    client.connect()
    result = client.segment(jpeg_bytes, prompts=["person"])
# session is automatically closed
```

## Decoding Masks

Each `Instance` has a `decode_mask()` method that returns a `(H, W)` boolean numpy array (requires the `[cv]` extra):

```python
for det in result.results:
    for inst in det.instances:
        mask = inst.decode_mask()  # numpy bool array (H, W)
```

## Webcam Demo

A full live-segmentation example with mask overlay is included in [`examples/webcam.py`](examples/webcam.py):

```bash
pip install "vizion[cv] @ git+https://github.com/CoreVisionX/vizion-sdk.git"
export VIZION_API_KEY="vz_live_..."
python examples/webcam.py
```

## Other Methods

```python
# List available models and pricing (no auth required)
models = client.models()
for m in models.models:
    print(f"{m.id}: {m.name} — {m.description}")
print(f"Cost: {models.cost_per_second_cents} cents/s")

# List your recent sessions
sessions = client.sessions()
for s in sessions:
    print(f"{s.id}  {s.status}  {s.duration_seconds}s")
```

## Response Types

All methods return typed [Pydantic](https://docs.pydantic.dev) models with full autocomplete support:

| Method | Return Type |
|---|---|
| `segment()` | `SegmentationResult` |
| `models()` | `ModelsResponse` |
| `sessions()` | `list[Session]` |

Key models:

- **`SegmentationResult`** — `results: list[Detection]`, plus timing fields (`decode_ms`, `vision_encode_ms`, `text_encode_ms`, `decode_segment_ms`)
- **`Detection`** — `prompt: str`, `instances: list[Instance]`
- **`Instance`** — `x1`, `y1`, `x2`, `y2`, `confidence`, `mask_rle`, `mask_height`, `mask_width`, plus `decode_mask()`
- **`ModelsResponse`** — `models: list[ModelInfo]`, `cost_per_second_cents: float`
- **`Session`** — `id`, `model`, `status`, `started_at`, `ended_at`, `duration_seconds`, `credits_used`

All models can be imported directly:

```python
from vizion import SegmentationResult, Detection, Instance, ModelsResponse, Session
```

## License

MIT
