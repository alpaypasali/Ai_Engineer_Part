import csv
import tempfile
from pathlib import Path
from src.predict import ReviewLens

engine = ReviewLens()

try:
    engine.predict("")
except ValueError:
    print("Empty Input: PASSED")
else:
    raise AssertionError("Empty input was accepted.")

long_text = "The food was delicious and the service was excellent. " * 80
long_text += "However, I found a piece of glass in my meal."

long_result = engine.predict(long_text)

tokenizer = engine.classifier.tokenizer

original_tokens = tokenizer(
    long_text,
    truncation=False,
    verbose=False
)["input_ids"]

truncated_tokens = tokenizer(
    long_text,
    truncation=True,
    max_length=256
)["input_ids"]

assert len(truncated_tokens) == 256

print("\nLong Review:", long_result)
print("Original Tokens:", len(original_tokens))
print("Truncated Tokens:", len(truncated_tokens))

texts = [
    "Amazing food.",
    "Terrible service.",
    "The food was average."
]

batch_results = engine.predict_batch(texts)

print("\nBatch Predictions:", batch_results)

ironic_text = "Great, another forty minutes waiting for cold food. Just perfect."

ironic_result = engine.predict(ironic_text)

print("\nIrony Prediction:", ironic_result)

mixed_text = "The food was excellent, but the staff were rude and the service was slow."

mixed_result = engine.predict(mixed_text)

print("\nMixed Sentiment:", mixed_result)

all_results = [
    long_result,
    *batch_results,
    ironic_result,
    mixed_result
]

assert all(
    1 <= result["predicted_stars"] <= 5
    for result in all_results
)

print("\nTest 1 — Star Range: PASSED")

assert all(
    0 <= result["confidence"] <= 1
    for result in all_results
)

print("Test 2 — Confidence Range: PASSED")

assert len(batch_results) == len(texts)

print("Test 3 — Batch Inference: PASSED")

engine.load_corpus("data/retrieval_corpus.csv")

similar_reviews = engine.search_similar("The food was delicious.")

assert len(similar_reviews) <= 5

print("Test 4 — Top-5 Retrieval: PASSED")

for text in ["", "   "]:
    try:
        engine.predict(text)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid input accepted.")

print("Test 5 — Input Validation: PASSED")

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "small_corpus.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "text", "stars"])
        writer.writerows([
            [10, "Delicious meals and friendly staff.", 5],
            [20, "Cold soup and rude service.", 1]
        ])

    engine.load_corpus(path)

    results = engine.search_similar("Delicious food.")

    assert len(results) == 2

print("Test 6 — Small Corpus: PASSED")
print("\nAll Tests: PASSED")