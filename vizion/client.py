"""Vizion.fast WebSocket client for real-time GPU segmentation."""

from __future__ import annotations

import json
import struct
import sys
import threading
import time
from typing import Any

import requests
import websockets.sync.client as ws_client

from vizion.models import DepthResult, ModelsResponse, SegmentationResult, Session

DEFAULT_API_URL = "https://www.vizion.fast"


class VizionClient:
    """Client for the Vizion.fast segmentation API.

    Handles authentication, session lifecycle, and the binary WebSocket
    protocol for sending JPEG frames and receiving segmentation results.

    Usage::

        from vizion import VizionClient

        client = VizionClient("vz_live_your_api_key")
        client.connect()

        result = client.segment(jpeg_bytes, prompts=["person", "car"])
        for det in result.results:
            print(det.prompt, len(det.instances))

        client.close()

    Or as a context manager::

        with VizionClient("vz_live_your_api_key") as client:
            client.connect()
            result = client.segment(jpeg_bytes, prompts=["person"])
    """

    def __init__(
        self,
        api_key: str,
        *,
        api_url: str = DEFAULT_API_URL,
        model: str = "sam3",
    ) -> None:
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.model = model
        self._ws: ws_client.ClientConnection | None = None
        self.session_id: str | None = None

    # -- context manager --------------------------------------------------

    def __enter__(self) -> VizionClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # -- session lifecycle ------------------------------------------------

    def connect(self, *, timeout: int = 180) -> None:
        """Start a GPU session and open the WebSocket connection.

        Calls ``POST /api/v1/connect`` to authenticate and provision a
        dedicated GPU worker, then connects to the returned WebSocket URL.

        Args:
            timeout: Maximum seconds to wait for the worker to start.

        Raises:
            PermissionError: Invalid API key or insufficient balance.
            ValueError: Invalid model.
            TimeoutError: Worker did not start in time.
        """
        url = f"{self.api_url}/api/v1/connect"
        print(f"Starting Vizion session (model={self.model}) ...")

        stop = threading.Event()

        def _ticker() -> None:
            start = time.time()
            while not stop.is_set():
                elapsed = time.time() - start
                sys.stdout.write(f"\r  Waiting for GPU worker ... {elapsed:.0f}s")
                sys.stdout.flush()
                stop.wait(2)
            sys.stdout.write("\r" + " " * 50 + "\r")
            sys.stdout.flush()

        t = threading.Thread(target=_ticker, daemon=True)
        t.start()

        try:
            resp = requests.post(
                url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.model},
                timeout=timeout,
            )
        finally:
            stop.set()
            t.join()

        if resp.status_code == 401:
            raise PermissionError(
                resp.json().get("error", "Invalid API key")
            )
        if resp.status_code == 402:
            raise PermissionError(
                resp.json().get("error", "Insufficient balance")
            )
        if resp.status_code == 400:
            raise ValueError(resp.json().get("error", "Bad request"))
        if resp.status_code == 504:
            raise TimeoutError(
                resp.json().get("error", "Session startup timed out")
            )
        resp.raise_for_status()

        data = resp.json()
        self.session_id = data["session_id"]
        ws_url: str = data["ws_url"]

        print(f"Session {self.session_id} ready — connecting to worker ...")
        self._ws = ws_client.connect(ws_url, max_size=10 * 1024 * 1024)
        print("Connected!")

    def segment(
        self,
        jpeg_bytes: bytes,
        prompts: list[str],
        *,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> SegmentationResult:
        """Send a JPEG frame and return segmentation results.

        Args:
            jpeg_bytes: Raw JPEG image bytes.
            prompts: Text prompts describing objects to segment
                (e.g. ``["person", "car"]``).
            score_threshold: Minimum confidence to keep a detection (0-1).
            mask_threshold: Threshold for binarising the mask (0-1).

        Returns:
            A :class:`~vizion.models.SegmentationResult` with typed access
            to detections, bounding boxes, masks, and server-side timing.

        Raises:
            RuntimeError: If :meth:`connect` has not been called.
        """
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")

        header = json.dumps(
            {
                "prompts": prompts,
                "score_threshold": score_threshold,
                "mask_threshold": mask_threshold,
            }
        ).encode()

        message = struct.pack("<I", len(header)) + header + jpeg_bytes
        self._ws.send(message)
        return SegmentationResult.model_validate_json(self._ws.recv())

    def depth(self, jpeg_bytes: bytes) -> DepthResult:
        """Send a JPEG frame and return a metric depth map.

        The depth server expects raw JPEG bytes (no header framing).

        Args:
            jpeg_bytes: Raw JPEG image bytes.

        Returns:
            A :class:`~vizion.models.DepthResult` with the base64-encoded
            uint16 PNG depth map and server-side timing.

        Raises:
            RuntimeError: If :meth:`connect` has not been called.
        """
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() first")

        self._ws.send(jpeg_bytes)
        return DepthResult.model_validate_json(self._ws.recv())

    def close(self) -> None:
        """Shut down the GPU worker and close the WebSocket.

        Sends the ``"shutdown"`` command so billing stops immediately.
        Safe to call multiple times.
        """
        if self._ws is not None:
            try:
                self._ws.send("shutdown")
            except Exception:
                pass
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            if self.session_id:
                print(f"Session {self.session_id} closed.")

    # -- REST helpers (no active session needed) --------------------------

    def models(self) -> ModelsResponse:
        """List available models and pricing.

        No authentication required.

        Returns:
            A :class:`~vizion.models.ModelsResponse` with model info and pricing.
        """
        resp = requests.get(f"{self.api_url}/api/v1/models", timeout=10)
        resp.raise_for_status()
        return ModelsResponse.model_validate(resp.json())

    def sessions(self) -> list[Session]:
        """List your recent sessions.

        Returns:
            A list of :class:`~vizion.models.Session` objects.
        """
        resp = requests.get(
            f"{self.api_url}/api/v1/sessions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        return [Session.model_validate(s) for s in resp.json()["sessions"]]
