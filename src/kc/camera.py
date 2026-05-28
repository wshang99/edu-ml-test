from __future__ import annotations

import cv2
import numpy as np


class Camera:
    """Webcam capture with an optional live preview window.

    Press Q in the preview window to close it.
    """

    def __init__(self, device: int = 0, width: int = 640, height: int = 480) -> None:
        self.device = device
        self.width = width
        self.height = height
        self._cap: cv2.VideoCapture | None = None
        self._preview_open = False
        self._window_name = "kc camera"
        self._overlay_text: str | None = None

    def _ensure_open(self) -> cv2.VideoCapture:
        if self._cap is None:
            cap = cv2.VideoCapture(self.device, cv2.CAP_ANY)
            if not cap.isOpened():
                raise RuntimeError(
                    f"Couldn't open webcam (device {self.device}). "
                    f"Check that no other program is using the camera, "
                    f"and that your operating system has given Python permission to use it."
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap = cap
        return self._cap

    def read(self) -> np.ndarray:
        """Grab one frame as a numpy array (BGR)."""
        cap = self._ensure_open()
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Couldn't read a frame from the webcam.")
        return frame

    def show(self) -> None:
        """Open a live preview window. Press Q to close."""
        self._preview_open = True
        self._ensure_open()

    def set_overlay(self, text: str | None) -> None:
        """Called by predict_stream to draw the current prediction onto the preview."""
        self._overlay_text = text

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if not self._overlay_text:
            return frame
        cv2.putText(
            frame,
            self._overlay_text,
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self._overlay_text,
            (12, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return frame

    def _pump_preview(self, frame: np.ndarray) -> bool:
        """Called per frame. Returns False if the user closed the window."""
        if not self._preview_open:
            return True
        cv2.imshow(self._window_name, self._draw_overlay(frame))
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            self.close()
            return False
        try:
            if cv2.getWindowProperty(self._window_name, cv2.WND_PROP_VISIBLE) < 1:
                self.close()
                return False
        except cv2.error:
            return True
        return True

    def __iter__(self):
        while True:
            frame = self.read()
            if not self._pump_preview(frame):
                return
            yield frame

    def close(self) -> None:
        if self._preview_open:
            try:
                cv2.destroyWindow(self._window_name)
            except cv2.error:
                pass
            self._preview_open = False
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
