import argparse
import csv
import json
from pathlib import Path

import numpy as np


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MAX_LENGTH = 256
BATCH_SIZE = 8
LABEL_MAPPING = {"1 star": 1, "2 stars": 2, "3 stars": 3, "4 stars": 4, "5 stars": 5}


def model_label_to_stars(label):
    if not isinstance(label, str) or label not in LABEL_MAPPING:
        raise ValueError(f"Invalid model label: {label}")
    return LABEL_MAPPING[label]


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Review must be a non-empty string.")
    return text


class ReviewLens:
    def __init__(self, device=None, classifier=None, embedding_model=None):
        self.device = device
        self.classifier = classifier
        self.embedding_model = embedding_model
        self.corpus = []
        self.corpus_embeddings = None

    def _classifier(self):
        if self.classifier is None:
            import torch
            from transformers import pipeline

            device = self.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            self.classifier = pipeline(
                "text-classification", model=MODEL_NAME, device=device,
                model_kwargs={"use_safetensors": True}, trust_remote_code=False
            )
            if set(self.classifier.model.config.id2label.values()) != set(LABEL_MAPPING):
                self.classifier = None
                raise ValueError("Unexpected classifier labels.")
        return self.classifier

    def predict(self, text):
        return self.predict_batch([validate_text(text)])[0]

    def predict_batch(self, texts):
        if not isinstance(texts, (list, tuple)) or not texts:
            raise ValueError("Provide a non-empty list of reviews.")
        texts = [validate_text(text) for text in texts]
        predictions = self._classifier()(
            texts, truncation=True, max_length=MAX_LENGTH, batch_size=BATCH_SIZE
        )
        if len(predictions) != len(texts):
            raise RuntimeError("Prediction count does not match input count.")
        results = []
        for prediction in predictions:
            confidence = float(prediction["score"])
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("Invalid confidence score.")
            results.append({"predicted_stars": model_label_to_stars(prediction["label"]), "confidence": confidence})
        return results

    def _encode(self, texts):
        if self.embedding_model is None:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(
                EMBEDDING_MODEL, device=self.device, trust_remote_code=False,
                model_kwargs={"use_safetensors": True}
            )
            self.embedding_model.max_seq_length = MAX_LENGTH
        vectors = np.asarray(self.embedding_model.encode(
            texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ))
        if vectors.shape != (len(texts), 384) or not np.isfinite(vectors).all():
            raise ValueError("Unexpected embedding shape or non-finite values.")
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
            raise ValueError("Expected unit-normalized embeddings.")
        return vectors

    def load_corpus(self, path):
        rows = []
        with Path(path).open(encoding="utf-8-sig", newline="") as file:
            reader = csv.DictReader(file)
            if not {"text", "stars"}.issubset(reader.fieldnames or []):
                raise ValueError("Corpus CSV must contain text and stars columns.")
            for index, row in enumerate(reader):
                raw_id = row.get("row_id", str(index))
                row_id = int(raw_id)
                stars = int(row["stars"])
                if stars not in range(1, 6):
                    raise ValueError("Corpus stars must be between 1 and 5.")
                rows.append({"row_id": row_id, "text": validate_text(row["text"]), "stars": stars})
        if not rows or len({row["row_id"] for row in rows}) != len(rows):
            raise ValueError("Corpus must contain at least one review with unique row IDs.")
        embeddings = self._encode([row["text"] for row in rows])
        self.corpus = rows
        self.corpus_embeddings = embeddings

    def search_similar(self, text):
        text = validate_text(text)
        if self.corpus_embeddings is None:
            raise RuntimeError("Load the corpus before searching.")
        query = self._encode([text])[0]
        scores = np.clip(self.corpus_embeddings @ query, -1, 1)
        indices = np.argsort(-scores, kind="stable")[:5]
        return [{**self.corpus[i], "similarity": float(scores[i])} for i in indices]

    def analyze(self, text):
        return {**self.predict(text), "similar_reviews": self.search_similar(text)}


def main():
    parser = argparse.ArgumentParser(description="ReviewLens star prediction and embedding retrieval")
    parser.add_argument("texts", nargs="+", help="One or more reviews")
    parser.add_argument("--corpus", type=Path, help="Local review CSV for Top-5 retrieval")
    parser.add_argument("--device", default=None, help="cpu or cuda:0; automatic by default")
    args = parser.parse_args()
    engine = ReviewLens(device=args.device)
    results = engine.predict_batch(args.texts)
    if args.corpus is not None:
        engine.load_corpus(args.corpus)
        for text, result in zip(args.texts, results):
            result["similar_reviews"] = engine.search_similar(text)
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
