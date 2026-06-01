"""MoveNet (single-pose, Lightning, int8) helpers.

MoveNet's TFLite output is a [1, 1, 17, 3] tensor: 17 keypoints in standard
COCO order, each row is (y, x, confidence) in normalized [0, 1] coordinates
relative to the model's 192x192 input window. This module maps that back to
original-frame pixel coordinates, swaps to (x, y, confidence), and labels
each keypoint by body-part name.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


# Standard COCO 17-keypoint order — matches MoveNet's output indexing exactly.
KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye", "right_eye",
    "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
)


# Skeleton edges drawn on the preview window when a PoseModel is active.
# Each entry is a pair of keypoint names that should be connected by a line.
SKELETON_EDGES: tuple[tuple[str, str], ...] = (
    ("nose", "left_eye"), ("nose", "right_eye"),
    ("left_eye", "left_ear"), ("right_eye", "right_ear"),
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
)


def weights_path() -> str:
    """Filesystem path to the bundled MoveNet Lightning int8 TFLite weights."""
    return str(Path(__file__).parent / "_weights" / "movenet_lightning_int8.tflite")


def preprocess(frame: np.ndarray, model_h: int, model_w: int) -> np.ndarray:
    """A BGR uint8 webcam frame -> (1, H, W, 3) uint8 RGB ready for MoveNet."""
    resized = cv2.resize(frame, (model_w, model_h))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0).astype(np.uint8)


def decode(
    raw_output: np.ndarray,
    frame_h: int,
    frame_w: int,
) -> dict[str, tuple[float, float, float]]:
    """MoveNet's raw [1, 1, 17, 3] output -> {body_part: (x_px, y_px, confidence)}."""
    kp = raw_output.reshape(17, 3)
    out: dict[str, tuple[float, float, float]] = {}
    for i, name in enumerate(KEYPOINT_NAMES):
        y_norm = float(kp[i, 0])
        x_norm = float(kp[i, 1])
        conf = float(kp[i, 2])
        out[name] = (x_norm * frame_w, y_norm * frame_h, conf)
    return out


def mean_confidence(keypoints: dict[str, tuple[float, float, float]]) -> float:
    """Average confidence across all 17 keypoints. Used as the Prediction's overall confidence."""
    if not keypoints:
        return 0.0
    return float(np.mean([v[2] for v in keypoints.values()]))
