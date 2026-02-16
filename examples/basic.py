"""Basic example: segment a single JPEG image."""

import os

from vizion import VizionClient

API_KEY = os.environ["VIZION_API_KEY"]

with VizionClient(API_KEY) as client:
    client.connect()

    # Load a JPEG image
    with open("frame.jpg", "rb") as f:
        jpeg_bytes = f.read()

    # Run segmentation
    result = client.segment(jpeg_bytes, prompts=["person", "car"])

    # Print results
    for det in result.results:
        n = len(det.instances)
        print(f"  {det.prompt}: {n} instance(s)")
        for inst in det.instances:
            print(
                f"    bbox=({inst.x1},{inst.y1})-({inst.x2},{inst.y2})  "
                f"confidence={inst.confidence:.2f}"
            )

    # Print server-side timing
    print(f"\nTiming: decode={result.decode_ms:.1f}ms  "
          f"vision={result.vision_encode_ms:.1f}ms  "
          f"text={result.text_encode_ms:.1f}ms  "
          f"segment={result.decode_segment_ms:.1f}ms")
