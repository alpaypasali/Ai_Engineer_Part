from math import isfinite


THRESHOLDS = (0.70, 0.80, 0.90, 0.95, 0.99)
DEFAULT_REVIEW_THRESHOLD = 0.80


def validate_probability(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return float(value)


def apply_decision_policy(prediction, threshold=DEFAULT_REVIEW_THRESHOLD):
    threshold = validate_probability(threshold, "threshold")
    confidence = validate_probability(prediction["confidence"], "confidence")
    reasons = []

    if confidence < threshold:
        reasons.append("low_confidence")
    if prediction["truncated"]:
        reasons.append("truncated_input")

    return {
        **prediction,
        "requires_review": bool(reasons),
        "review_reasons": reasons,
    }


def analyze_thresholds(predictions, actual_labels, thresholds=THRESHOLDS):
    if not predictions or len(predictions) != len(actual_labels):
        raise ValueError("Predictions and labels must have the same non-zero length")

    total = len(predictions)
    rows = []

    for threshold in thresholds:
        accepted = [
            prediction["label"] == actual
            for prediction, actual in zip(predictions, actual_labels)
            if not apply_decision_policy(prediction, threshold)["requires_review"]
        ]
        count = len(accepted)
        rows.append({
            "threshold": threshold,
            "coverage": count / total,
            "accepted_prediction_accuracy": sum(accepted) / count if count else None,
            "rejected_samples": total - count,
        })

    return rows


def select_threshold(rows, target_accuracy=0.98):
    target_accuracy = validate_probability(target_accuracy, "target_accuracy")
    eligible = [
        row for row in rows
        if row["accepted_prediction_accuracy"] is not None
        and row["accepted_prediction_accuracy"] >= target_accuracy
    ]
    return max(eligible, key=lambda row: (row["coverage"], -row["threshold"]), default=None)
