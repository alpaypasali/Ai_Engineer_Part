import argparse
from pathlib import Path
import joblib

from preprocessing import clean_text


# ==========================================
# 1. MODEL CONFIGURATION
# ==========================================

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "complaintrouter_lr.joblib"

model = None


# ==========================================
# 2. MODEL LOADING
# ==========================================

def load_model():
    global model

    # Model daha önce yüklenmediyse yükle
    if model is None:

        # Model dosyası mevcut mu?
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Model file not found: {MODEL_PATH}"
            )

        # Modeli yükle
        model = joblib.load(MODEL_PATH)

    return model


# ==========================================
# 3. CATEGORY MAPPING
# ==========================================

category_mapping = {
    0: "Credit reporting, credit repair services, or other personal consumer reports",
    1: "Debt collection",
    2: "Credit reporting",
    3: "Mortgage"
}


# ==========================================
# 4. PREDICTION FUNCTION
# ==========================================

def predict_complaint(text):

    # --------------------------------------
    # STEP 1: INPUT VALIDATION
    # --------------------------------------

    if not isinstance(text, str):
        raise TypeError(
            "Complaint text must be a string."
        )

    if not text.strip():
        raise ValueError(
            "Complaint text cannot be empty."
        )

    # --------------------------------------
    # STEP 2: TEXT PREPROCESSING
    # --------------------------------------

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError(
            "Complaint text is empty after cleaning."
        )

    # --------------------------------------
    # STEP 3: LOAD MODEL
    # --------------------------------------

    model = load_model()

    # --------------------------------------
    # STEP 4: MAKE PREDICTION
    # --------------------------------------

    # Pipeline kendi preprocessing işlemini yapar.
    # Bu nedenle ham metni gönderiyoruz.

    prediction = model.predict([text])[0]

    # --------------------------------------
    # STEP 5: PREDICTION PROBABILITIES
    # --------------------------------------

    probabilities = model.predict_proba([text])[0]

    # Tahmin edilen sınıfın indeksini bul
    class_index = list(model.classes_).index(prediction)

    # Tahmin edilen sınıfın olasılığı
    confidence = round(
        float(probabilities[class_index]),
        2
    )

    # --------------------------------------
    # STEP 6: HUMAN REVIEW
    # --------------------------------------

    if confidence >= 0.60:
        status = "automatic"
    else:
        status = "human review required"

    # --------------------------------------
    # STEP 7: CATEGORY NAME
    # --------------------------------------

    category = category_mapping.get(
        int(prediction),
        "Unknown"
    )

    # --------------------------------------
    # STEP 8: RETURN RESULT
    # --------------------------------------

    result = {
        "category": category,
        "probability": confidence,
        "status": status
    }

    return result


# ==========================================
# 5. TERMINAL APPLICATION
# ==========================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="ComplaintRouter AI"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Complaint text to classify"
    )

    args = parser.parse_args()

    sample_text = args.text

    try:

        result = predict_complaint(sample_text)

        print("\nComplaintRouter AI")
        print("-------------------------------")

        print(
            f"Category: {result['category']}"
        )

        print(
            f"Probability: {result['probability']:.2f}"
        )

        print(
            f"Status: {result['status'].title()}"
        )

    except (ValueError, TypeError, FileNotFoundError) as e:

        print(f"Error: {e}")