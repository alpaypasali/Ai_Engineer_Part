import csv
import tempfile
from pathlib import Path

from src.predict import ReviewLens


def main():
    engine = ReviewLens(device="cpu")
    texts = ["The meal was delicious and the staff were friendly.", "The food was cold and the service was awful."]
    single = engine.predict(texts[0])
    batch = engine.predict_batch(texts + ["The food was delicious. " * 400])
    assert len(batch) == 3
    assert single["predicted_stars"] == batch[0]["predicted_stars"]
    assert all(1 <= row["predicted_stars"] <= 5 and 0 <= row["confidence"] <= 1 for row in batch)
    print("Real classification (single, batch, long input): PASSED")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "corpus.csv"
        corpus = [
            ("Wonderful meals and friendly waiters.", 5),
            ("I waited an hour for cold food.", 1),
            ("There was a large selection of seafood.", 4),
            ("The coffee was bitter.", 2),
            ("A pleasant dinner with excellent service.", 5),
            ("The restaurant was noisy.", 3),
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["row_id", "text", "stars"])
            for i, (text, stars) in enumerate(corpus):
                writer.writerow([i, text, stars])
        engine.load_corpus(path)
        result = engine.analyze(texts[0])
        matches = result["similar_reviews"]
        assert len(matches) == len({row["row_id"] for row in matches}) == 5
        assert all(-1 <= row["similarity"] <= 1 for row in matches)
        assert all(a["similarity"] >= b["similarity"] for a, b in zip(matches, matches[1:]))
        print("Real embedding retrieval (Top-5): PASSED")
        print(result)


if __name__ == "__main__":
    main()
