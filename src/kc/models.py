from __future__ import annotations

import os

from kc.errors import ModelLoadError, ModelTypeError
from kc.result import Prediction


class _BaseModel:
    """Shared loader logic for TFLite-backed models."""

    _model_kind: str = "image"

    def __init__(self, path: str) -> None:
        self.path = path
        self.labels: list[str] = []
        self._interpreter = None
        self._load(path)

    def _load(self, path: str) -> None:
        if not os.path.isfile(path):
            raise ModelLoadError.file_missing(path)
        if not path.lower().endswith(".tflite"):
            raise ModelLoadError.not_a_tflite_file(path)
        # TODO: real load via ai_edge_litert.Interpreter, sniff model kind,
        # raise ModelTypeError.wrong_class(...) if mismatched, load labels.txt.
        raise NotImplementedError(
            "TFLite loading is not wired yet. "
            "Next step: ai_edge_litert.Interpreter + labels.txt parsing."
        )

    def predict(self, frame) -> Prediction:
        raise NotImplementedError(
            f"{type(self).__name__}.predict() is not wired yet. "
            f"Frame received: shape={getattr(frame, 'shape', '?')}."
        )


class ImageModel(_BaseModel):
    """Image classifier (Teachable Machine 'Image Project', or a built-in pretrained)."""

    _model_kind = "image"

    def __init__(self, path_or_name: str) -> None:
        if path_or_name == "mobilenet":
            # TODO: download/cache MobileNet TFLite and load.
            raise NotImplementedError("Built-in 'mobilenet' is not wired yet.")
        super().__init__(path_or_name)


class PoseModel(_BaseModel):
    """Pose classifier (Teachable Machine 'Pose Project'), or a pretrained pose model.

    Predictions include result.keypoints — a dict of body part names to (x, y, confidence).
    """

    _model_kind = "pose"

    def __init__(self, path_or_name: str) -> None:
        if path_or_name == "movenet":
            # TODO: ship MoveNet TFLite weights, load + expose keypoints only
            # (no classification head — labels stay empty).
            raise NotImplementedError("Built-in 'movenet' is not wired yet.")
        super().__init__(path_or_name)
