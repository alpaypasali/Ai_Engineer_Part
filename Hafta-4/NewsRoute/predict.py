import sys
import re
import json
import joblib

from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models/tfidf_pipeline.joblib"
RARE_WORDS_PATH = BASE_DIR / "models/rare_words.json"

LABEL_MAP = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

model = joblib.load(MODEL_PATH)

with open(RARE_WORDS_PATH, "r", encoding="utf-8") as f:
    rare_words = set(json.load(f))

sw = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^\w\s]", "", text)

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = [
        word for word in text.split()
        if word not in sw
    ]

    tokens = [
        word for word in tokens
        if word not in rare_words
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return " ".join(tokens)



def validate_text(text):

    if not isinstance(text, str):
        raise TypeError("Haber metni string olmalıdır.")

    if not text.strip():
        raise ValueError("Haber metni boş olamaz.")

    return text


def predict_news(text):

    text = validate_text(text)

    cleaned_text = clean_text(text)

    if not cleaned_text:
        raise ValueError("Temizleme sonrasında haber metni boş kaldı.")

    prediction = model.predict([cleaned_text])[0]

    return LABEL_MAP[int(prediction)]


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print('Usage: python predict.py "News text"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    try:
        category = predict_news(text)

        print(f"\nInput: {text}")
        print(f"Predicted Class: {category}")

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)