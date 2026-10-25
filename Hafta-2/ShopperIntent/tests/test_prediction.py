import json
import joblib
import numpy as np
import pytest
import torch

from pathlib import Path

from src.serving import (
    ShopperClassifier,
    load_model,
    predict_purchase
)


# ==========================================
# FIXTURES
# ==========================================

@pytest.fixture(scope="module")
def loaded_components():

    # Test boyunca modeli bir kez yükle
    return load_model()


@pytest.fixture
def valid_session():

    project_root = Path(__file__).resolve().parent.parent

    sample_path = (
        project_root / "samples" / "session_01.json"
    )

    with open(sample_path, "r", encoding="utf-8") as f:
        session_data = json.load(f)

    return session_data


# ==========================================
# TEST 1 — VALID SESSION
# ==========================================

def test_valid_session(
    valid_session,
    loaded_components
):

    model, preprocessor, config = loaded_components

    result = predict_purchase(
        valid_session,
        model,
        preprocessor,
        config
    )

    expected_keys = {
        "purchase_probability",
        "predicted_revenue",
        "threshold",
        "target_campaign",
        "model_version"
    }

    assert isinstance(result, dict)

    assert set(result.keys()) == expected_keys

    assert 0 <= result["purchase_probability"] <= 1

    assert isinstance(result["predicted_revenue"], bool)

    assert isinstance(result["target_campaign"], bool)

    assert result["threshold"] == 0.60


# ==========================================
# TEST 2 — MISSING COLUMN
# ==========================================

def test_missing_column(
    valid_session,
    loaded_components
):

    model, preprocessor, config = loaded_components

    # Month sütununu kaldır
    valid_session.pop("Month")

    with pytest.raises(
        ValueError,
        match="Missing features"
    ):

        predict_purchase(
            valid_session,
            model,
            preprocessor,
            config
        )


# ==========================================
# TEST 3 — INVALID NUMERIC VALUE
# ==========================================

def test_invalid_numeric_value(
    valid_session,
    loaded_components
):

    model, preprocessor, config = loaded_components

    valid_session["ProductRelated_Duration"] = "hello"

    with pytest.raises(
        TypeError,
        match="Invalid numeric value"
    ):

        predict_purchase(
            valid_session,
            model,
            preprocessor,
            config
        )


# ==========================================
# TEST 4 — NAN / INF
# ==========================================

def test_nan_inf(
    valid_session,
    loaded_components
):

    model, preprocessor, config = loaded_components

    invalid_values = [
        float("nan"),
        float("inf")
    ]

    for invalid_value in invalid_values:

        session = valid_session.copy()

        session["ProductRelated_Duration"] = invalid_value

        with pytest.raises(
            ValueError,
            match="NaN or Inf"
        ):

            predict_purchase(
                session,
                model,
                preprocessor,
                config
            )


# ==========================================
# TEST 5 — MODEL FILE NOT FOUND
# ==========================================

def test_model_file_not_found(monkeypatch):

  
    original_is_file = Path.is_file

    def fake_is_file(path):

        if path.name == "shopper_mlp.pt":
            return False

        return original_is_file(path)

    monkeypatch.setattr(
        Path,
        "is_file",
        fake_is_file
    )

    with pytest.raises(
        FileNotFoundError,
        match="shopper_mlp.pt"
    ):

        load_model()


# ==========================================
# TEST 6 — SERIALIZATION CONSISTENCY
# ==========================================

def test_serialization_consistency(
    valid_session,
    loaded_components,
    tmp_path
):

    original_model, original_preprocessor, config = loaded_components

    original_result = predict_purchase(
        valid_session,
        original_model,
        original_preprocessor,
        config
    )


    model_path = tmp_path / "test_model.pt"

    torch.save(
        original_model.state_dict(),
        model_path
    )

    preprocessor_path = tmp_path / "test_preprocessor.joblib"

    joblib.dump(
        original_preprocessor,
        preprocessor_path
    )


    loaded_model = ShopperClassifier(
        activation=config["activation"],
        input_dim=config["input_dim"]
    )


    loaded_model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )

    loaded_model.eval()

    loaded_preprocessor = joblib.load(
        preprocessor_path
    )


    loaded_result = predict_purchase(
        valid_session,
        loaded_model,
        loaded_preprocessor,
        config
    )

    original_probability = original_result[
        "purchase_probability"
    ]

    loaded_probability = loaded_result[
        "purchase_probability"
    ]

 
    assert np.isclose(
        original_probability,
        loaded_probability,
        atol=1e-6,
        rtol=0
    )


    assert (
        original_result["predicted_revenue"]
        == loaded_result["predicted_revenue"]
    )

    assert (
        original_result["target_campaign"]
        == loaded_result["target_campaign"]
    )