from __future__ import annotations

from typing import Iterator

from kc.camera import Camera
from kc.models import _BaseModel
from kc.result import Prediction


def predict_stream(camera: Camera, model: _BaseModel) -> Iterator[Prediction]:
    """Yield one Prediction per webcam frame.

    Iterating this also pumps the camera preview (if camera.show() was called)
    and draws the top label onto the live frame.
    """
    for frame in camera:
        result = model.predict(frame)
        camera.set_overlay(str(result))
        yield result
