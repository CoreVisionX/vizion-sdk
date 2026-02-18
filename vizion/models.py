"""Pydantic response models for the Vizion API."""

from __future__ import annotations

from pydantic import BaseModel


class Instance(BaseModel):
    """A single detected object instance with bounding box and RLE mask."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    mask_rle: list[int]
    mask_height: int
    mask_width: int

    def decode_mask(self) -> "np.ndarray":
        """Decode the RLE mask into a ``(H, W)`` boolean numpy array.

        Requires numpy (install with ``pip install vizion[cv]``).
        """
        import numpy as np

        flat = np.zeros(self.mask_height * self.mask_width, dtype=bool)
        pos = 0
        for i, length in enumerate(self.mask_rle):
            if i % 2 == 1:
                flat[pos : pos + length] = True
            pos += length
        return flat.reshape((self.mask_height, self.mask_width), order="F")


class Detection(BaseModel):
    """Results for a single text prompt."""

    prompt: str
    instances: list[Instance]


class SegmentationResult(BaseModel):
    """Full response from a ``segment()`` call."""

    results: list[Detection]
    decode_ms: float
    vision_encode_ms: float
    text_encode_ms: float
    decode_segment_ms: float


class ModelInfo(BaseModel):
    """Description of an available model."""

    id: str
    name: str
    description: str


class ModelsResponse(BaseModel):
    """Response from the ``models()`` endpoint."""

    cost_per_second_cents: float
    models: list[ModelInfo]


class DepthResult(BaseModel):
    """Full response from a ``depth()`` call.

    The depth map is returned as a base64-encoded uint16 PNG.  Use
    :meth:`decode_depth` to convert it to a float32 numpy array with
    metric depth values (in metres).
    """

    depth_png_b64: str
    depth_min: float
    depth_max: float
    height: int
    width: int
    decode_ms: float
    inference_ms: float
    encode_ms: float

    def decode_depth(self) -> "np.ndarray":
        """Decode the PNG into a ``(H, W)`` float32 depth map in metres.

        Requires numpy and Pillow (install with ``pip install vizion[cv]``).
        """
        import base64
        import io

        import numpy as np
        from PIL import Image

        png_bytes = base64.b64decode(self.depth_png_b64)
        img = Image.open(io.BytesIO(png_bytes))
        depth_u16 = np.array(img, dtype=np.float32)
        # Reverse uint16 normalisation → metric depth
        depth = depth_u16 / 65535.0 * (self.depth_max - self.depth_min) + self.depth_min
        return depth


class Session(BaseModel):
    """A session record from the ``sessions()`` endpoint."""

    id: str
    model: str
    status: str
    started_at: str
    ended_at: str | None = None
    duration_seconds: float | None = None
    credits_used: int | None = None
