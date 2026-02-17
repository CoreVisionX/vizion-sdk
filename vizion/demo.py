"""Live webcam demo: real-time segmentation with mask overlay.

Usage::

    python -m vizion.demo --key vz_live_xxxxx

Requires: pip install vizion[cv]
"""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Vizion live webcam demo")
    parser.add_argument(
        "--key",
        default=os.environ.get("VIZION_API_KEY"),
        help="Vizion API key (or set VIZION_API_KEY env var)",
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["person"],
        help="Text prompts to segment (default: person)",
    )
    args = parser.parse_args()

    if not args.key:
        print("Error: pass --key or set VIZION_API_KEY", file=sys.stderr)
        sys.exit(1)

    try:
        import cv2
        import numpy as np
    except ImportError:
        print(
            'Error: opencv and numpy required. Install with:\n'
            '  pip install "vizion[cv]"',
            file=sys.stderr,
        )
        sys.exit(1)

    from vizion import VizionClient

    COLORS = [
        (0, 255, 0),
        (255, 0, 0),
        (0, 0, 255),
        (255, 255, 0),
    ]

    with VizionClient(args.key) as client:
        client.connect()

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: could not open webcam", file=sys.stderr)
            sys.exit(1)

        print("Press 'q' to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue

            result = client.segment(buf.tobytes(), prompts=args.prompts)

            overlay = frame.copy()
            for i, det in enumerate(result.results):
                color = COLORS[i % len(COLORS)]
                for inst in det.instances:
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


if __name__ == "__main__":
    main()
