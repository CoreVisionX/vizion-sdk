"""Webcam demo: live segmentation with mask overlay.

Requires: pip install vizion[cv]
"""

import os

import cv2
import numpy as np

from vizion import VizionClient

API_KEY = os.environ["VIZION_API_KEY"]
PROMPTS = ["person"]

# Assign a colour per prompt
COLORS = [
    (0, 255, 0),
    (255, 0, 0),
    (0, 0, 255),
    (255, 255, 0),
]

with VizionClient(API_KEY) as client:
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

        result = client.segment(buf.tobytes(), prompts=PROMPTS)

        # Draw masks and bounding boxes
        overlay = frame.copy()
        for i, det in enumerate(result.results):
            color = COLORS[i % len(COLORS)]
            for inst in det.instances:
                # Draw mask
                mask = inst.decode_mask()
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.uint8),
                        (frame.shape[1], frame.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                overlay[mask] = (
                    overlay[mask] * 0.5 + np.array(color) * 0.5
                ).astype(np.uint8)

                # Draw bounding box
                cv2.rectangle(
                    overlay,
                    (int(inst.x1), int(inst.y1)),
                    (int(inst.x2), int(inst.y2)),
                    color,
                    2,
                )
                cv2.putText(
                    overlay,
                    f"{det.prompt} {inst.confidence:.0%}",
                    (int(inst.x1), int(inst.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                )

        cv2.imshow("Vizion", overlay)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
