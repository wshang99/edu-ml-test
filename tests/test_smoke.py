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


def test_pose_model_movenet_is_a_stub_with_helpful_message():
    from kc import PoseModel

    with pytest.raises(NotImplementedError) as exc:
        PoseModel("movenet")
    assert "movenet" in str(exc.value).lower()


def test_real_tflite_filename_hits_the_unimplemented_loader(tmp_path):
    """Confirms the stub is reached when path looks right but TFLite load isn't wired."""
    from kc import ImageModel

    fake = tmp_path / "image_model.tflite"
    fake.write_bytes(b"\x00" * 32)  # exists, ends in .tflite — passes both early checks
    with pytest.raises(NotImplementedError) as exc:
        ImageModel(str(fake))
    # the stub message must point to what to wire next, not be a blind NotImplementedError
    assert "ai_edge_litert" in str(exc.value) or "Interpreter" in str(exc.value)


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
