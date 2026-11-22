from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.predict import FinanceSentinel


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "financesentinel_final"
LABELS = {"Negative", "Neutral", "Positive"}
TEXTS = [
    "The company increased its operating profit by 15%.",
    "Net losses widened significantly during the quarter.",
    "The company will publish its quarterly report next Tuesday.",
]


@pytest.fixture(scope="session")
def predictor():
    if not MODEL_PATH.is_dir():
        pytest.fail(f"Final model directory not found: {MODEL_PATH}")
    return FinanceSentinel(model_path=MODEL_PATH, device="cpu", batch_size=2)


@pytest.fixture(scope="session")
def prediction(predictor):
    return predictor.predict(TEXTS[0])


def test_model_loading(predictor):
    assert predictor.model is not None
    assert predictor.tokenizer is not None
    assert not predictor.model.training
    assert predictor.model.config.num_labels == 3
    assert predictor.max_length == 256


def test_single_prediction(prediction):
    assert set(prediction) == {
        "label", "confidence", "probabilities", "truncated",
        "requires_review", "review_reasons",
    }
    assert isinstance(prediction["label"], str)
    assert isinstance(prediction["requires_review"], bool)
    assert prediction["truncated"] is False


def test_batch_prediction(predictor):
    results = predictor.predict_batch(TEXTS, batch_size=2)
    assert len(results) == len(TEXTS)
    for batch_size in [0, -1, True, 1.5]:
        with pytest.raises((TypeError, ValueError)):
            predictor.predict_batch(TEXTS, batch_size=batch_size)


def test_empty_input(predictor):
    for text in ["", "   ", "\n\t"]:
        with pytest.raises(ValueError):
            predictor.predict(text)
    with pytest.raises(ValueError):
        predictor.predict_batch([])
    with pytest.raises(ValueError):
        predictor.predict_batch([TEXTS[0], ""])


def test_invalid_input_type(predictor):
    for text in [123, None, 3.14, ["text"], {"text": "profit"}]:
        with pytest.raises(TypeError):
            predictor.predict(text)
    with pytest.raises(TypeError):
        predictor.predict_batch(TEXTS[0])
    with pytest.raises(TypeError):
        predictor.predict_batch([TEXTS[0], 123])


def test_long_input(predictor):
    text = "profit " * 300
    inputs, flags = predictor.preprocess([text])
    assert inputs["input_ids"].shape[1] == 256
    assert flags == [True]
    result = predictor.predict(text)
    assert result["truncated"] is True
    assert result["requires_review"] is True
    assert "truncated_input" in result["review_reasons"]


def test_probability_sum(prediction):
    probabilities = prediction["probabilities"]
    assert len(probabilities) == 3
    assert all(0.0 <= value <= 1.0 for value in probabilities.values())
    assert sum(probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_confidence_validity(prediction):
    confidence = prediction["confidence"]
    assert 0.0 <= confidence <= 1.0
    assert confidence == pytest.approx(max(prediction["probabilities"].values()), abs=1e-6)
    assert confidence == pytest.approx(prediction["probabilities"][prediction["label"]], abs=1e-6)


def test_label_validity(predictor):
    assert set(predictor.id2label.values()) == LABELS
    results = predictor.predict_batch(TEXTS)
    assert all(result["label"] in LABELS for result in results)


def test_prediction_consistency(predictor):
    single_results = [predictor.predict(text) for text in TEXTS]
    batch_results = predictor.predict_batch(TEXTS, batch_size=2)
    for single, batch in zip(single_results, batch_results, strict=True):
        assert single["label"] == batch["label"]
        assert single["confidence"] == pytest.approx(batch["confidence"], abs=1e-5)
        assert single["probabilities"] == pytest.approx(batch["probabilities"], abs=1e-5)
        assert single["truncated"] == batch["truncated"]
        assert single["requires_review"] == batch["requires_review"]
        assert single["review_reasons"] == batch["review_reasons"]


def test_threshold_policy(predictor, monkeypatch):
    assert predictor.review_threshold == pytest.approx(0.80)

    def uncertain_forward(**inputs):
        logits = torch.zeros((inputs["input_ids"].shape[0], 3), device=inputs["input_ids"].device)
        return SimpleNamespace(logits=logits)

    monkeypatch.setattr(predictor.model, "forward", uncertain_forward)
    result = predictor.predict(TEXTS[0])
    assert result["confidence"] == pytest.approx(1 / 3)
    assert result["truncated"] is False
    assert result["requires_review"] is True
    assert "low_confidence" in result["review_reasons"]


def test_model_reusability(predictor, monkeypatch):
    model = predictor.model
    tokenizer = predictor.tokenizer

    def forbidden_reload(*args, **kwargs):
        pytest.fail("Model or tokenizer was reloaded during prediction")

    monkeypatch.setattr("src.predict.AutoModelForSequenceClassification.from_pretrained", forbidden_reload)
    monkeypatch.setattr("src.predict.AutoTokenizer.from_pretrained", forbidden_reload)
    predictor.predict(TEXTS[0])
    predictor.predict_batch(TEXTS)
    assert predictor.model is model
    assert predictor.tokenizer is tokenizer
