from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from kc.errors import ModelLoadError
from kc.result import Prediction


# ---------- helpers ----------

def _read_labels(model_path: str) -> list[str]:
    """Read a labels.txt sitting next to the model file.

    Teachable Machine exports include this alongside model.tflite. Two common formats
    are handled: "0 cat" (TM's default) and bare "cat" (one label per line).
    Missing file -> empty list (predictions fall back to "class_<n>").
    """
    labels_path = os.path.join(os.path.dirname(os.path.abspath(model_path)), "labels.txt")
    if not os.path.isfile(labels_path):
        return []
    out: list[str] = []
    with open(labels_path, "r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0].isdigit():
                out.append(parts[1])
            else:
                out.append(line)
    return out


def _preprocess_image(frame: np.ndarray, input_shape, dtype) -> np.ndarray:
    """Resize a webcam (BGR uint8) frame for a TM image model.

    - Resize to model's H x W.
    - Convert BGR -> RGB.
    - For float models: normalize to [-1, 1] (MobileNet/TM convention).
    - For quantized (uint8) models: pass raw bytes through.
    """
    _, h, w, _ = tuple(input_shape)
    resized = cv2.resize(frame, (int(w), int(h)))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    batched = np.expand_dims(rgb, axis=0)
    if np.issubdtype(dtype, np.floating):
        return ((batched.astype(np.float32) / 127.5) - 1.0).astype(dtype)
    return batched.astype(dtype)


# ---------- model classes ----------

class _BaseModel:
    """Shared loader for TFLite-backed models."""

    _model_kind: str = "image"

    def __init__(self, path: str) -> None:
        self.path = path
        self.labels: list[str] = []
        self._interpreter: Any = None
        self._input_details: list[dict] | None = None
        self._output_details: list[dict] | None = None
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise ModelLoadError.file_missing(path)
        if not path.lower().endswith(".tflite"):
            raise ModelLoadError.not_a_tflite_file(path)
        try:
            from ai_edge_litert.interpreter import Interpreter
            interp = Interpreter(model_path=path)
            interp.allocate_tensors()
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError.not_a_tflite_file(path) from exc
        self._interpreter = interp
        self._input_details = interp.get_input_details()
        self._output_details = interp.get_output_details()
        self.labels = _read_labels(path)

    @property
    def interpreter(self) -> Any:
        """Escape hatch: the underlying ai_edge_litert.Interpreter for advanced users."""
        return self._interpreter

    def predict(self, frame: np.ndarray) -> Prediction:
        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not wired yet."
        )


class ImageModel(_BaseModel):
    """Image classifier — Teachable Machine 'Image Project' export, or a built-in pretrained."""

    _model_kind = "image"

    def __init__(self, path_or_name: str) -> None:
        if path_or_name == "mobilenet":
            raise NotImplementedError(
                "Built-in 'mobilenet' is not wired yet. "
                "For now, train an image model on Teachable Machine and pass its .tflite path."
            )
        super().__init__(path_or_name)

    def predict(self, frame: np.ndarray) -> Prediction:
        in_det = self._input_details[0]
        out_det = self._output_details[0]

        x = _preprocess_image(frame, in_det["shape"], in_det["dtype"])
        self._interpreter.set_tensor(in_det["index"], x)
        self._interpreter.invoke()
        raw = self._interpreter.get_tensor(out_det["index"])[0]

        # Dequantize if needed (uint8 output models).
        q = out_det.get("quantization", (0.0, 0))
        if q and q[0] != 0:
            scale, zero_point = q
            scores = (raw.astype(np.float32) - zero_point) * scale
        else:
            scores = raw.astype(np.float32)

        top = int(np.argmax(scores))
        label = self.labels[top] if top < len(self.labels) else f"class_{top}"
        scores_dict = {
            (self.labels[i] if i < len(self.labels) else f"class_{i}"): float(s)
            for i, s in enumerate(scores)
        }
        return Prediction(label=label, confidence=float(scores[top]), scores=scores_dict)


class PoseModel(_BaseModel):
    """Pose classifier (TM Pose Project) or pretrained MoveNet.

    Predictions include result.keypoints — a dict of body part names to (x, y, confidence).
    """

    _model_kind = "pose"

    def __init__(self, path_or_name: str) -> None:
        if path_or_name == "movenet":
            raise NotImplementedError(
                "Built-in 'movenet' is not wired yet. "
                "v0.0.1 wires image classification only; pose comes next."
            )
        super().__init__(path_or_name)

    def predict(self, frame: np.ndarray) -> Prediction:
        raise NotImplementedError(
            "PoseModel.predict() is not wired yet — v0.0.1 only wires ImageModel. "
            "Keypoint decoding and the MoveNet builtin land next."
        )
