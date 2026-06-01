"""Smoke + contract tests for the current skeleton.

These tests don't need a webcam, GPU, or any real model file. They lock in:
  - the public API surface
  - the Prediction dataclass behavior
  - the contract on every user-facing error message
  - the stub paths (so accidental "real" code in a stub fails loudly)
"""

from __future__ import annotations

import pytest


# ---------- public API surface ----------

def test_kc_module_imports():
    import kc  # noqa: F401


def test_version_is_a_string():
    import kc

    assert isinstance(kc.__version__, str)
    assert kc.__version__.count(".") >= 1


def test_all_public_names_present():
    import kc

    expected = {
        "Camera",
        "ImageModel",
        "PoseModel",
        "Prediction",
        "predict_stream",
        "KcError",
        "ModelLoadError",
        "ModelTypeError",
    }
    assert set(kc.__all__) == expected
    for name in expected:
        assert hasattr(kc, name), f"kc.{name} missing"


def test_error_hierarchy_lets_users_catch_one_class():
    from kc import KcError, ModelLoadError, ModelTypeError

    assert issubclass(ModelLoadError, KcError)
    assert issubclass(ModelTypeError, KcError)


# ---------- Prediction dataclass ----------

def test_prediction_defaults():
    from kc import Prediction

    p = Prediction(label="cat", confidence=0.9)
    assert p.label == "cat"
    assert p.confidence == 0.9
    assert p.scores == {}
    assert p.keypoints is None


def test_prediction_is_confident_uses_default_threshold():
    from kc import Prediction

    assert Prediction(label="cat", confidence=0.9).is_confident is True
    assert Prediction(label="cat", confidence=0.6).is_confident is True
    assert Prediction(label="cat", confidence=0.3).is_confident is False


def test_prediction_str_is_kid_friendly():
    from kc import Prediction

    s = str(Prediction(label="dancing", confidence=0.87))
    assert "dancing" in s
    assert "87" in s  # rendered as a percentage


def test_prediction_can_carry_keypoints():
    from kc import Prediction

    p = Prediction(
        label="pose_1",
        confidence=0.9,
        keypoints={"nose": (320.0, 240.0, 0.95)},
    )
    assert p.keypoints is not None
    assert p.keypoints["nose"] == (320.0, 240.0, 0.95)


# ---------- error message contracts (load-bearing for lesson UX) ----------

def test_model_load_error_for_wrong_extension_mentions_TM_and_TFLite():
    from kc import ModelLoadError

    msg = str(ModelLoadError.not_a_tflite_file("dance.h5"))
    assert "Teachable Machine" in msg
    assert "TensorFlow Lite" in msg
    assert "dance.h5" in msg


def test_model_load_error_for_missing_file_mentions_the_filename():
    from kc import ModelLoadError

    msg = str(ModelLoadError.file_missing("my_model.tflite"))
    assert "my_model.tflite" in msg
    # nudge kids toward looking in the right folder
    assert "folder" in msg.lower() or "script" in msg.lower()


def test_model_type_error_names_the_right_class_to_try_instead():
    from kc import ModelTypeError

    msg = str(ModelTypeError.wrong_class("dance.tflite", "pose", "PoseModel"))
    assert "dance.tflite" in msg
    assert "PoseModel" in msg
    assert "pose" in msg


# ---------- model class behavior (current stub state) ----------

def test_image_model_raises_friendly_error_for_missing_file():
    from kc import ImageModel, ModelLoadError

    with pytest.raises(ModelLoadError) as exc:
        ImageModel("definitely_not_a_real_file.tflite")
    assert "definitely_not_a_real_file.tflite" in str(exc.value)


def test_image_model_raises_friendly_error_for_non_tflite_file(tmp_path):
    from kc import ImageModel, ModelLoadError

    bogus = tmp_path / "model.h5"
    bogus.write_bytes(b"not a tflite file")
    with pytest.raises(ModelLoadError) as exc:
        ImageModel(str(bogus))
    assert "Teachable Machine" in str(exc.value)


def test_image_model_mobilenet_is_a_stub_with_helpful_message():
    from kc import ImageModel

    with pytest.raises(NotImplementedError) as exc:
        ImageModel("mobilenet")
    assert "mobilenet" in str(exc.value).lower()


def test_pose_model_movenet_loads_and_predicts_on_synthetic_frame():
    """End-to-end: bundled MoveNet weights load, predict runs, 17 named keypoints come back
    in pixel coords of the original frame with sensible confidences."""
    import numpy as np

    from kc import PoseModel, Prediction
    from kc.movenet import KEYPOINT_NAMES

    model = PoseModel("movenet")

    # Synthetic 480x640 BGR uint8 frame — no real person; just checking the wiring.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = model.predict(frame)

    assert isinstance(result, Prediction)
    assert result.label == "pose"
    assert 0.0 <= result.confidence <= 1.0
    assert result.keypoints is not None
    # all 17 COCO names present, no extras
    assert set(result.keypoints.keys()) == set(KEYPOINT_NAMES)
    # each entry is (x, y, confidence); coords inside frame; conf in [0, 1]
    for name, (x, y, c) in result.keypoints.items():
        assert 0.0 <= x <= 640, f"{name} x={x} out of frame width"
        assert 0.0 <= y <= 480, f"{name} y={y} out of frame height"
        assert 0.0 <= c <= 1.0, f"{name} conf={c} out of [0,1]"


def test_movenet_weights_file_is_bundled():
    """The bundled .tflite must ship with the package — otherwise PoseModel('movenet') breaks."""
    import os

    from kc.movenet import weights_path

    p = weights_path()
    assert os.path.isfile(p), f"MoveNet weights not found at {p}"
    # Sanity: file is non-trivially sized (real model is ~2-3 MB)
    assert os.path.getsize(p) > 1_000_000


def test_movenet_module_exports_skeleton_and_names():
    from kc.movenet import KEYPOINT_NAMES, SKELETON_EDGES

    assert len(KEYPOINT_NAMES) == 17
    assert "nose" in KEYPOINT_NAMES
    # every skeleton edge references real keypoint names
    for a, b in SKELETON_EDGES:
        assert a in KEYPOINT_NAMES
        assert b in KEYPOINT_NAMES


def test_malformed_tflite_file_raises_friendly_load_error(tmp_path):
    """A .tflite path that exists but isn't a valid TFLite model should surface as ModelLoadError,
    not as a raw flatbuffer/runtime exception from ai_edge_litert."""
    from kc import ImageModel, ModelLoadError

    fake = tmp_path / "image_model.tflite"
    fake.write_bytes(b"\x00" * 32)  # passes the existence + extension checks
    with pytest.raises(ModelLoadError) as exc:
        ImageModel(str(fake))
    assert "Teachable Machine" in str(exc.value)


def test_pose_model_with_malformed_path_raises_load_error(tmp_path):
    """For non-movenet paths (e.g. a TM Pose Project export), load runs first.
    A malformed file should surface the TM-aware ModelLoadError, same as ImageModel."""
    from kc import PoseModel, ModelLoadError

    fake = tmp_path / "pose_model.tflite"
    fake.write_bytes(b"\x00" * 32)
    with pytest.raises(ModelLoadError):
        PoseModel(str(fake))


def test_image_model_exposes_raw_interpreter_via_property():
    """The .interpreter escape hatch is part of the documented advanced surface."""
    from kc import ImageModel

    # Property exists at class level even without an instance.
    assert hasattr(ImageModel, "interpreter")
    assert isinstance(ImageModel.interpreter, property)


# ---------- Camera (no webcam required) ----------

def test_camera_constructs_without_opening_hardware():
    from kc import Camera

    cam = Camera()
    assert cam.device == 0
    assert cam.width == 640
    assert cam.height == 480
    cam.close()  # no-op when nothing's open


def test_camera_accepts_custom_resolution():
    from kc import Camera

    cam = Camera(device=1, width=1280, height=720)
    assert cam.device == 1
    assert cam.width == 1280
    assert cam.height == 720


def test_camera_set_overlay_stores_text_without_opening_hardware():
    from kc import Camera

    cam = Camera()
    cam.set_overlay("dancing 87%")
    assert cam._overlay_text == "dancing 87%"
    cam.set_overlay(None)
    assert cam._overlay_text is None


def test_camera_context_manager_protocol():
    from kc import Camera

    with Camera() as cam:
        assert cam is not None
    # exiting context should not raise even if hardware was never opened


# ---------- predict_stream wiring ----------

def test_predict_stream_is_importable_and_callable():
    from kc import predict_stream

    assert callable(predict_stream)
