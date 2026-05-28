"""kc — machine learning for Python beginners."""

from kc.camera import Camera
from kc.errors import KcError, ModelLoadError, ModelTypeError
from kc.models import ImageModel, PoseModel
from kc.predict import predict_stream
from kc.result import Prediction

__version__ = "0.0.1"

__all__ = [
    "Camera",
    "ImageModel",
    "PoseModel",
    "Prediction",
    "predict_stream",
    "KcError",
    "ModelLoadError",
    "ModelTypeError",
]
