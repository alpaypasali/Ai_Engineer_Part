from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __package__:
    from .decision_policy import DEFAULT_REVIEW_THRESHOLD, apply_decision_policy, validate_probability
    from .cli import run_cli
else:
    from decision_policy import DEFAULT_REVIEW_THRESHOLD, apply_decision_policy, validate_probability
    from cli import run_cli


class FinanceSentinel:
    def __init__(self, model_path=None, device=None, batch_size=16, review_threshold=DEFAULT_REVIEW_THRESHOLD):
        self.max_length = 256
        self.batch_size = self._validate_batch_size(batch_size)

        self.review_threshold = validate_probability(review_threshold, "review_threshold")

        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "financesentinel_final"

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True, use_safetensors=True
        ).to(self.device)
        self.model.eval()

        self.id2label = {int(key): value for key, value in self.model.config.id2label.items()}
        if self.model.config.num_labels != 3 or set(self.id2label) != {0, 1, 2}:
            raise ValueError("The final model must have three configured sentiment classes")
        if len(set(self.id2label.values())) != 3:
            raise ValueError("Model label names must be unique")

    @staticmethod
    def _validate_batch_size(batch_size):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return batch_size

    @staticmethod
    def validate_input(text):
        if not isinstance(text, str):
            raise TypeError("Input must be a string")

        text = text.strip()

        if not text:
            raise ValueError("Input cannot be empty")

        return text

    def _validate_texts(self, texts):
        if not isinstance(texts, (list, tuple)):
            raise TypeError("Batch input must be a list or tuple of strings")
        if not texts:
            raise ValueError("Batch input cannot be empty")
        return [self.validate_input(text) for text in texts]

    def preprocess(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        texts = self._validate_texts(texts)

        original = self.tokenizer(
            texts, padding=False, truncation=False, add_special_tokens=True,
            return_token_type_ids=False, return_attention_mask=False, verbose=False
        )

        truncated = [len(ids) > self.max_length for ids in original["input_ids"]]

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length,
            return_tensors="pt", return_token_type_ids=False
        ).to(self.device)

        return inputs, truncated

    def predict(self, text):
        return self.predict_batch([self.validate_input(text)], batch_size=1)[0]

    def predict_batch(self, texts, batch_size=None):
        texts = self._validate_texts(texts)
        size = self.batch_size if batch_size is None else self._validate_batch_size(batch_size)
        results = []

        for i in range(0, len(texts), size):
            inputs, truncated = self.preprocess(texts[i:i + size])

            with torch.inference_mode():
                logits = self.model(**inputs).logits
                probabilities = torch.softmax(logits, dim=1)
                confidence, predictions = probabilities.max(dim=1)

            for row, score, prediction, was_truncated in zip(
                probabilities.cpu().tolist(), confidence.cpu().tolist(),
                predictions.cpu().tolist(), truncated
            ):
                result = {
                    "label": self.id2label[prediction],
                    "confidence": score,
                    "probabilities": {
                        self.id2label[class_id]: probability
                        for class_id, probability in enumerate(row)
                    },
                    "truncated": was_truncated,
                }
                results.append(apply_decision_policy(result, self.review_threshold))

        return results


if __name__ == "__main__":
    raise SystemExit(run_cli(FinanceSentinel))
