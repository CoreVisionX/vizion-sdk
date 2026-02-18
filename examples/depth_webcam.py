"""Webcam demo: live depth estimation with colourmap overlay.

Requires: pip install vizion[cv]
"""

import os

import cv2
import numpy as np

from vizion import VizionClient

API_KEY = os.environ["VIZION_API_KEY"]

with VizionClient(API_KEY, model="depth-anything-3") as client:
    client.connect()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Encode frame as JPEG
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            continue

        result = client.depth(buf.tobytes())
        depth = result.decode_depth()  # (H, W) float32 metres

        # Normalise to 0-255 for visualisation
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 0:
            norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
        else:
            norm = np.zeros_like(depth, dtype=np.uint8)

        # Apply colourmap (closer = warm, farther = cool)
        depth_color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)

        # Resize depth to match frame if needed
        if depth_color.shape[:2] != frame.shape[:2]:
            depth_color = cv2.resize(depth_color, (frame.shape[1], frame.shape[0]))

        # Side-by-side display
        combined = np.hstack([frame, depth_color])
        cv2.putText(
            combined,
            f"{result.inference_ms:.0f}ms",
            (frame.shape[1] + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        cv2.imshow("Vizion Depth", combined)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
