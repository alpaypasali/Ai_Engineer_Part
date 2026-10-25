import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import joblib
import math

from numbers import Real
from pathlib import Path

from .decision_policy import campaign_decision


# ==========================================
# 1. MODEL ARCHITECTURE
# ==========================================

class ShopperClassifier(nn.Module):

    def __init__(self, activation="relu", input_dim=73):
        super().__init__()

        if activation == "relu":
            activation_layer = nn.ReLU

        elif activation == "gelu":
            activation_layer = nn.GELU

        else:
            raise ValueError(
                "Activation must be 'relu' or 'gelu'"
            )

        self.linear_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            activation_layer(),
            nn.Linear(64, 32),
            activation_layer(),
            nn.Linear(32, 2),
        )

    def forward(self, x):
        return self.linear_layer(x)


# ==========================================
# 2. LOAD MODEL
# ==========================================

def load_model():

    project_root = Path(__file__).resolve().parent.parent
    models_dir = project_root / "models"

 
    config_path = models_dir / "model_config.json"

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

  
    preprocessor_path = models_dir / "preprocessor.joblib"

    
    preprocessor = joblib.load(preprocessor_path)

  
    model_path = models_dir / "shopper_mlp.pt"

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model weights file not found: {model_path}"
        )

   
    model = ShopperClassifier(
        activation=config["activation"],
        input_dim=config["input_dim"]
    )

    model.load_state_dict(
        torch.load(
            model_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()

    return model, preprocessor, config


# ==========================================
# 3. PREDICT PURCHASE
# ==========================================

def predict_purchase(
    session_data,
    model,
    preprocessor,
    config
):

    # --------------------------------------
    # 1. INPUT VALIDATION
    # --------------------------------------

    if not isinstance(session_data, dict):
        raise TypeError(
            "session_data must be a dictionary."
        )

    feature_columns = config["feature_columns"]

  
    forbidden_columns = ["PageValues", "Revenue"]

    invalid_columns = [
        col for col in forbidden_columns
        if col in session_data
    ]

    if invalid_columns:
        raise ValueError(
            f"Forbidden features: {invalid_columns}"
        )

    missing_columns = [
        col for col in feature_columns
        if col not in session_data
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    extra_columns = [
        col for col in session_data
        if col not in feature_columns
    ]

    if extra_columns:
        raise ValueError(
            f"Unexpected features: {extra_columns}"
        )

    # --------------------------------------
    # 2. MISSING / NAN / INF CONTROL
    # --------------------------------------

    for column, value in session_data.items():

        if value is None:
            raise ValueError(
                f"Missing value in '{column}'."
            )

        if isinstance(value, Real) and not isinstance(value, bool):

            if not math.isfinite(float(value)):
                raise ValueError(
                    f"Non-finite value in '{column}': "
                    "NaN or Inf is not allowed."
                )

    # --------------------------------------
    # 3. NUMERIC TYPE VALIDATION
    # --------------------------------------

    categorical_columns = [
        "Month",
        "VisitorType",
        "Weekend"
    ]

    numeric_columns = [
        col for col in feature_columns
        if col not in categorical_columns
    ]

    for column in numeric_columns:

        value = session_data[column]

        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Invalid numeric value for '{column}': "
                "expected int or float."
            )

    # --------------------------------------
    # 4. DATAFRAME
    # --------------------------------------

    session_df = pd.DataFrame(
        [session_data],
        columns=feature_columns
    )

    if session_df.isna().any().any():
        raise ValueError(
            "Session data contains missing values."
        )

    # --------------------------------------
    # 5. PREPROCESSING
    # --------------------------------------

    # Tahmin sırasında FIT işlemi yapılmaz.
    transformed_data = preprocessor.transform(
        session_df
    )

    if hasattr(transformed_data, "toarray"):
        transformed_data = transformed_data.toarray()

    transformed_data = np.asarray(
        transformed_data,
        dtype=np.float32
    )

    if not np.isfinite(transformed_data).all():
        raise ValueError(
            "Preprocessed data contains NaN or Inf."
        )

    if transformed_data.shape[1] != config["input_dim"]:
        raise ValueError(
            "Preprocessing output dimension does not match model input."
        )

    # --------------------------------------
    # 6. TENSOR CONVERSION
    # --------------------------------------

    input_tensor = torch.tensor(
        transformed_data,
        dtype=torch.float32
    )

    # --------------------------------------
    # 7. PREDICTION
    # --------------------------------------

    model.eval()

    with torch.no_grad():

        logits = model(input_tensor)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        positive_class = config["positive_class"]

        purchase_probability = probabilities[
            0, positive_class
        ].item()

    # --------------------------------------
    # 8. DECISION POLICY
    # --------------------------------------

    threshold = config["decision_threshold"]

    predicted_revenue = bool(
        purchase_probability >= 0.50
    )

    decision = campaign_decision(
        probability=purchase_probability,
        threshold=threshold
    )

    # --------------------------------------
    # 9. RESULT
    # --------------------------------------

    return {
        "purchase_probability": purchase_probability,
        "predicted_revenue": predicted_revenue,
        "threshold": threshold,
        "target_campaign": decision["target"],
        "model_version": config["model_version"]
    }