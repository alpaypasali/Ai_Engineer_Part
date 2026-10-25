import math
from numbers import Real


def campaign_decision(probability, threshold=0.50):

    # Veri tipi kontrolü
    if isinstance(probability, bool) or not isinstance(probability, Real):
        raise TypeError("Probability must be numeric.")

    if isinstance(threshold, bool) or not isinstance(threshold, Real):
        raise TypeError("Threshold must be numeric.")

    # NaN ve Infinity kontrolü
    if not math.isfinite(probability):
        raise ValueError("Probability must be finite.")

    if not math.isfinite(threshold):
        raise ValueError("Threshold must be finite.")

    # Aralık kontrolü
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    # Karar
    target = bool(probability >= threshold)

    return {
        "probability": float(probability),
        "threshold": float(threshold),
        "target": target
    }