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
        self._overlay_keypoints: dict | None = None

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
        """Called by predict_stream to set the top label rendered on the preview."""
        self._overlay_text = text

    def set_keypoints(self, keypoints: dict | None) -> None:
        """Called by predict_stream when a PoseModel is active. Skeleton auto-draws on the preview."""
        self._overlay_keypoints = keypoints

    def _draw_overlay(self, frame: np.ndarray) -> np.ndarray:
        if self._overlay_keypoints:
            self._draw_skeleton(frame, self._overlay_keypoints)
        if self._overlay_text:
            cv2.putText(
                frame, self._overlay_text, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA,
            )
            cv2.putText(
                frame, self._overlay_text, (12, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA,
            )
        return frame

    @staticmethod
    def _draw_skeleton(frame: np.ndarray, keypoints: dict) -> None:
        """Draw keypoint dots and connecting edges. Skips low-confidence points."""
        from kc.movenet import SKELETON_EDGES  # local import: keep camera independent of model code at import time

        conf_threshold = 0.3
        # edges
        for a, b in SKELETON_EDGES:
            if a not in keypoints or b not in keypoints:
                continue
            ax, ay, ac = keypoints[a]
            bx, by, bc = keypoints[b]
            if ac < conf_threshold or bc < conf_threshold:
                continue
            pa = (int(ax), int(ay))
            pb = (int(bx), int(by))
            cv2.line(frame, pa, pb, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.line(frame, pa, pb, (0, 255, 255), 2, cv2.LINE_AA)
        # joints
        for name, (x, y, c) in keypoints.items():
            if c < conf_threshold:
                continue
            p = (int(x), int(y))
            cv2.circle(frame, p, 5, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, p, 3, (255, 255, 255), -1, cv2.LINE_AA)

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
