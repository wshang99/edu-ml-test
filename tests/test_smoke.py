def test_public_api_imports():
    import kc

    assert hasattr(kc, "Camera")
    assert hasattr(kc, "ImageModel")
    assert hasattr(kc, "PoseModel")
    assert hasattr(kc, "predict_stream")
    assert hasattr(kc, "Prediction")
    assert hasattr(kc, "ModelLoadError")
    assert hasattr(kc, "ModelTypeError")


def test_prediction_is_confident_threshold():
    from kc import Prediction

    assert Prediction(label="cat", confidence=0.9).is_confident is True
    assert Prediction(label="cat", confidence=0.3).is_confident is False


def test_model_load_error_messages_mention_teachable_machine():
    from kc import ModelLoadError

    msg = str(ModelLoadError.not_a_tflite_file("foo.h5"))
    assert "Teachable Machine" in msg
    assert "TensorFlow Lite" in msg


def test_model_type_error_names_the_right_class():
    from kc import ModelTypeError

    msg = str(ModelTypeError.wrong_class("dance.tflite", "pose", "PoseModel"))
    assert "PoseModel" in msg
    assert "dance.tflite" in msg
