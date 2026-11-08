from datasets import load_dataset
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
import joblib
import json
import torch
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from transformer_model import NewsRouteTransformer


nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("stopwords", quiet=True)

lemmatizer = WordNetLemmatizer()
sw = set(stopwords.words("english"))


def normalize_text(text):

    text = text.fillna("").astype(str)

    text = text.str.lower()

    text = text.str.replace(r'[^\w\s]', '', regex=True)

    text = text.str.replace(r'\d+', '', regex=True)

    text = text.str.replace(r'\s+', ' ', regex=True).str.strip()

    text = text.apply(
        lambda x: " ".join(
            word for word in x.split()
            if word not in sw
        )
    )

    return text


dataset = load_dataset("fancyzhx/ag_news")


raw_sample_df = (
    dataset["train"].to_pandas()
    .groupby("label", group_keys=False)
    .sample(n=1000, random_state=42)
)

sample_normalized = normalize_text(raw_sample_df["text"])

word_counts = pd.Series(
    " ".join(sample_normalized).split()
).value_counts()

rare_words = set(
    word_counts[word_counts <= 1].index
)

Path("models/rare_words.json").write_text(
    json.dumps(sorted(rare_words), ensure_ascii=False),
    encoding="utf-8"
)

def clean_text(text):

    text = normalize_text(text)

    text = text.apply(
        lambda x: " ".join(
            word for word in x.split()
            if word not in rare_words
        )
    )

    text = text.apply(
        lambda x: " ".join(
            lemmatizer.lemmatize(word)
            for word in x.split()
        )
    )

    return text


print("\nPREPROCESSING")

print("Original Training Sample:", len(raw_sample_df))
print("Rare Words:", len(rare_words))


train_df = pd.read_csv("data/train_split.csv")
val_df = pd.read_csv("data/val_split.csv")

reconstructed_texts = clean_text(raw_sample_df["text"])

reconstructed_rows = Counter(
    zip(
        reconstructed_texts.tolist(),
        raw_sample_df["label"].tolist()
    )
)

saved_rows = Counter(
    zip(
        pd.concat(
            [train_df["text"], val_df["text"]],
            ignore_index=True
        ).tolist(),
        pd.concat(
            [train_df["label"], val_df["label"]],
            ignore_index=True
        ).tolist()
    )
)

assert reconstructed_rows == saved_rows, (
    "Preprocessing, kaydedilen train/validation verileriyle eşleşmiyor."
)

print("Training Preprocessing Match: True")


test_df = dataset["test"].to_pandas()

print("\nFINAL TEST DATA")

print("Test Samples:", len(test_df))
print("Columns:", test_df.columns.tolist())

print("\nClass Distribution:")
print(test_df["label"].value_counts().sort_index())

assert len(test_df) == 7600
assert test_df["label"].nunique() == 4

test_texts = clean_text(test_df["text"])

y_true = test_df["label"].to_numpy()

print("\nOriginal:")
print(test_df["text"].iloc[0])

print("\nCleaned:")
print(test_texts.iloc[0])


tfidf_model = joblib.load(
    "models/tfidf_pipeline.joblib"
)

tfidf_predictions = tfidf_model.predict(test_texts)

tfidf_accuracy = accuracy_score(
    y_true,
    tfidf_predictions
)

tfidf_precision = precision_score(
    y_true,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

tfidf_recall = recall_score(
    y_true,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

tfidf_f1 = f1_score(
    y_true,
    tfidf_predictions,
    average="macro",
    zero_division=0
)

print("\nTF-IDF — Final Test Results")

print("Predictions:", len(tfidf_predictions))
print(f"Accuracy: {tfidf_accuracy:.4f}")
print(f"Macro Precision: {tfidf_precision:.4f}")
print(f"Macro Recall: {tfidf_recall:.4f}")
print(f"Macro-F1: {tfidf_f1:.4f}")


with open(
    "models/vocab.json",
    "r",
    encoding="utf-8"
) as f:
    vocab = json.load(f)


MAX_LENGTH = 64
PAD_ID = 0
UNK_ID = 1

assert len(vocab) == 7787
assert vocab["[PAD]"] == PAD_ID
assert vocab["[UNK]"] == UNK_ID


transformer_model = NewsRouteTransformer(
    vocab_size=len(vocab),
    d_model=64,
    num_heads=4,
    num_layers=1,
    num_classes=4
)

transformer_model.load_state_dict(
    torch.load(
        "models/newsroute_transformer.pt",
        map_location="cpu",
        weights_only=True
    )
)

transformer_model.eval()


def encode_text(text):

    tokens = text.lower().split()

    token_ids = [
        vocab.get(token, UNK_ID)
        for token in tokens
    ]

    token_ids = token_ids[:MAX_LENGTH]

    token_ids += [PAD_ID] * (
        MAX_LENGTH - len(token_ids)
    )

    return token_ids


test_encoded = [
    encode_text(text)
    for text in test_texts
]

X_test = torch.tensor(
    test_encoded,
    dtype=torch.long
)

print("\nTRANSFORMER INPUT")

print("Vocabulary Size:", len(vocab))
print("Input Shape:", X_test.shape)


transformer_predictions = []

with torch.no_grad():

    for start in range(0, len(X_test), 32):

        batch = X_test[start:start + 32]

        logits = transformer_model(batch)

        assert torch.isfinite(logits).all().item()

        predictions = torch.argmax(
            logits,
            dim=1
        )

        transformer_predictions.extend(
            predictions.tolist()
        )


assert len(transformer_predictions) == 7600
assert len(tfidf_predictions) == 7600


transformer_accuracy = accuracy_score(
    y_true,
    transformer_predictions
)

transformer_precision = precision_score(
    y_true,
    transformer_predictions,
    average="macro",
    zero_division=0
)

transformer_recall = recall_score(
    y_true,
    transformer_predictions,
    average="macro",
    zero_division=0
)

transformer_f1 = f1_score(
    y_true,
    transformer_predictions,
    average="macro",
    zero_division=0
)

print("\nTransformer — Final Test Results")

print("Predictions:", len(transformer_predictions))
print(f"Accuracy: {transformer_accuracy:.4f}")
print(f"Macro Precision: {transformer_precision:.4f}")
print(f"Macro Recall: {transformer_recall:.4f}")
print(f"Macro-F1: {transformer_f1:.4f}")


final_metrics = pd.DataFrame([
    {
        "model": "TFIDF_LogisticRegression",
        "test_samples": len(y_true),
        "accuracy": tfidf_accuracy,
        "macro_precision": tfidf_precision,
        "macro_recall": tfidf_recall,
        "macro_f1": tfidf_f1
    },
    {
        "model": "NewsRouteTransformer",
        "test_samples": len(y_true),
        "accuracy": transformer_accuracy,
        "macro_precision": transformer_precision,
        "macro_recall": transformer_recall,
        "macro_f1": transformer_f1
    }
])


reports_dir = Path("reports")
reports_dir.mkdir(parents=True, exist_ok=True)

final_metrics.to_csv(
    reports_dir / "final_test_metrics.csv",
    index=False
)

print("\nFINAL MODEL COMPARISON")
print(final_metrics.round(4).to_string(index=False))

print("\nSaved: reports/final_test_metrics.csv")