"""Vizion SDK — real-time GPU segmentation over WebSocket."""

from vizion.client import VizionClient
from vizion.models import (
    Detection,
    Instance,
    ModelInfo,
    ModelsResponse,
    SegmentationResult,
    Session,
)

__all__ = [
    "VizionClient",
    "SegmentationResult",
    "Detection",
    "Instance",
    "ModelInfo",
    "ModelsResponse",
    "Session",
]
__version__ = "0.2.0"
