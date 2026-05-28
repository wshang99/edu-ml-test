from dataclasses import dataclass, field

DEFAULT_CONFIDENCE_THRESHOLD = 0.6


@dataclass
class Prediction:
    """One prediction from a model. Returned per frame by predict_stream."""

    label: str
    confidence: float
    scores: dict[str, float] = field(default_factory=dict)
    keypoints: dict[str, tuple[float, float, float]] | None = None

    @property
    def is_confident(self) -> bool:
        return self.confidence >= DEFAULT_CONFIDENCE_THRESHOLD

    def __str__(self) -> str:
        return f"{self.label} ({self.confidence:.0%})"
