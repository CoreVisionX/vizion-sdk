"""Basic example: estimate depth for a single JPEG image."""

import os

from vizion import VizionClient

API_KEY = os.environ["VIZION_API_KEY"]

with VizionClient(API_KEY, model="depth-anything-3") as client:
    client.connect()

    # Load a JPEG image
    with open("frame.jpg", "rb") as f:
        jpeg_bytes = f.read()

    # Run depth estimation
    result = client.depth(jpeg_bytes)

    print(f"Depth map: {result.width}x{result.height}")
    print(f"Range: {result.depth_min:.3f} – {result.depth_max:.3f} metres")

    # Decode to numpy array (requires vizion[cv])
    depth = result.decode_depth()
    print(f"Array: shape={depth.shape}, dtype={depth.dtype}")

    # Print server-side timing
    print(f"\nTiming: decode={result.decode_ms:.1f}ms  "
          f"inference={result.inference_ms:.1f}ms  "
          f"encode={result.encode_ms:.1f}ms")
