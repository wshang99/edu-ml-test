class KcError(Exception):
    """Base class for kc errors. Catch this to catch anything kc raises."""


class ModelLoadError(KcError):
    """Raised when a model file can't be read or isn't a Teachable Machine TFLite export."""

    @classmethod
    def not_a_tflite_file(cls, path: str) -> "ModelLoadError":
        return cls(
            f"Couldn't load '{path}'. This file doesn't look like a Teachable Machine "
            f"TFLite export. Did you select 'TensorFlow Lite' (not 'TensorFlow') when "
            f"downloading from Teachable Machine?"
        )

    @classmethod
    def file_missing(cls, path: str) -> "ModelLoadError":
        return cls(
            f"Couldn't find '{path}'. Make sure the model file is in the same folder "
            f"as your script, and the filename is spelled exactly right."
        )


class ModelTypeError(KcError):
    """Raised when a model file is loaded with the wrong class (e.g. a pose model into ImageModel)."""

    @classmethod
    def wrong_class(cls, path: str, actual: str, suggested_class: str) -> "ModelTypeError":
        return cls(
            f"'{path}' was trained as a {actual} model. "
            f"Try kc.{suggested_class}('{path}') instead."
        )
