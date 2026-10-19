import pytest
from predict import predict_complaint
import predict





@pytest.fixture
def fake_model(monkeypatch):

    class FakeModel:

        classes_ = [0, 1, 2, 3]

        def predict(self, texts):
            return [3]

        def predict_proba(self, texts):
            return [[0.05, 0.05, 0.10, 0.80]]

    monkeypatch.setattr(
        predict,
        "model",
        FakeModel()
    )

def test_empty_string():

    with pytest.raises(ValueError):
        predict_complaint("")



def test_whitespace_only():

    with pytest.raises(ValueError):
        predict_complaint("     ")


def test_none_input():

    with pytest.raises(TypeError):
        predict_complaint(None)

def test_valid_prediction_dictionary(fake_model):

    result = predict.predict_complaint(
        "My mortgage payment is incorrect"
    )

    assert isinstance(result, dict)

    assert set(result.keys()) == {
        "category",
        "probability",
        "status"
    }

    assert result["category"] == "Mortgage"

    assert result["status"] == "automatic"



def test_probability_between_zero_and_one(fake_model):

    result = predict.predict_complaint(
        "My mortgage payment is incorrect"
    )

    probability = result["probability"]

    assert 0 <= probability <= 1

def test_missing_model_file(monkeypatch, tmp_path):

    monkeypatch.setattr(
        predict,
        "model",
        None
    )

    fake_path = tmp_path / "missing.joblib"

    monkeypatch.setattr(
        predict,
        "MODEL_PATH",
        fake_path
    )

    with pytest.raises(
        FileNotFoundError,
        match="Model file not found"
    ):
        predict.predict_complaint(
            "My mortgage payment is incorrect"
        )