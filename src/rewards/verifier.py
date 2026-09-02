"""Binary verifiable reward for mathematical answers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .answer_parser import extract_final_answer, normalize_answer


@dataclass(frozen=True)
class VerificationResult:
    reward: float
    prediction: str | None
    ground_truth: str | None
    correct: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def verify_answer(completion: object, ground_truth: object) -> VerificationResult:
    prediction = extract_final_answer(completion)
    target = extract_final_answer(ground_truth)
    if target is None:
        target = normalize_answer(ground_truth)
    correct = prediction is not None and target is not None and prediction == target
    return VerificationResult(float(correct), prediction, target, correct)
