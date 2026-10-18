import argparse
from pathlib import Path
import joblib


from preprocessing import clean_text


# 1. Modeli yükle

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_PATH = PROJECT_ROOT / "models" / "complaintrouter_lr.joblib"


model = joblib.load(MODEL_PATH)


# 2. Kategori eşleştirmesi

category_mapping = {
    0: "Credit reporting, credit repair services, or other personal consumer reports",
    1: "Debt collection",
    2: "Credit reporting",
    3: "Mortgage"
}


# 3. Tahmin fonksiyonu
def predict_complaint(text):

    # GÖREV 3.1 — Input Validation
    if not isinstance(text, str):
        raise TypeError("Complaint text must be a string.")

    if not text.strip():
        raise ValueError("Complaint text cannot be empty.")

    # GÖREV 3.2 — Text Preprocessing
    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Complaint text is empty after cleaning.")

    # GÖREV 3.3 — Prediction
    # Pipeline temizleme işlemini kendisi yapar.
    prediction = model.predict([text])[0]

    # Prediction Probabilities
    probabilities = model.predict_proba([text])[0]

    # Tahmin edilen sınıfın olasılığını bul
    class_index = list(model.classes_).index(prediction)

    confidence = round(float(probabilities[class_index]), 2)

    status = "automatic" if probabilities[class_index] >= 0.60 else "human review required"

    # Kategori ismini belirle
    category = category_mapping.get(
    int(prediction),
    "Unknown"
    )
    # GÖREV 3.4 — Dictionary Result
    result = {
        "category": category,
        "probability": confidence,
        "status": status
    }

    return result

# 4. Terminal Test
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
        print(f"Category: {result['category']}")
        print(f"Probability: {result['probability']:.2f}")
        print(f"Status: {result['status'].title()}")

    except (ValueError, TypeError) as e:
        print(f"Error: {e}")
