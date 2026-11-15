import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.predict import ReviewLens, model_label_to_stars


class FakeClassifier:
    def __call__(self, texts, **kwargs):
        if kwargs != {"truncation": True, "max_length": 256, "batch_size": 8}:
            raise AssertionError(kwargs)
        return [{"label": "5 stars" if "good" in text else "1 star", "score": 0.8} for text in texts]


class FakeEncoder:
    max_seq_length = 256

    def encode(self, texts, **kwargs):
        if not kwargs["normalize_embeddings"]:
            raise AssertionError("Normalization required")
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        for i, text in enumerate(texts):
            angle = float(text.split()[0])
            vectors[i, :2] = [np.cos(angle), np.sin(angle)]
        return vectors


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.engine = ReviewLens(classifier=FakeClassifier(), embedding_model=FakeEncoder())

    def test_mapping(self):
        for label, stars in zip(["1 star", "2 stars", "3 stars", "4 stars", "5 stars"], range(1, 6)):
            self.assertEqual(model_label_to_stars(label), stars)
        with self.assertRaises(ValueError):
            model_label_to_stars("6 stars")

    def test_single_prediction(self):
        self.assertEqual(self.engine.predict("good food"), {"predicted_stars": 5, "confidence": 0.8})

    def test_batch_keeps_order(self):
        results = self.engine.predict_batch(["bad food", "good service"])
        self.assertEqual([r["predicted_stars"] for r in results], [1, 5])

    def test_empty_input(self):
        for value in ["", "   ", None]:
            with self.assertRaises(ValueError):
                self.engine.predict(value)
        with self.assertRaises(ValueError):
            self.engine.predict_batch([])
        with self.assertRaises(ValueError):
            self.engine.predict_batch("good")

    def test_retrieval_uses_descending_similarity_and_original_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corpus.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["row_id", "text", "stars"])
                for i, angle in enumerate([1, 0.2, 0.5, 0.1, 0.4, 0.3]):
                    writer.writerow([100 + i, f"{angle} review", 3])
            self.engine.load_corpus(path)
            results = self.engine.search_similar("0 query")
            self.assertEqual([r["row_id"] for r in results], [103, 101, 105, 104, 102])
            self.assertTrue(all(a["similarity"] >= b["similarity"] for a, b in zip(results, results[1:])))
            self.assertEqual(len(results), 5)

    def test_retrieval_requires_corpus(self):
        with self.assertRaises(RuntimeError):
            self.engine.search_similar("0 query")

