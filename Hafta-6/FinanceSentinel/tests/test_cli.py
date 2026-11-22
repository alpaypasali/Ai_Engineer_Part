import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from src.cli import run_cli
from src.decision_policy import apply_decision_policy


class FakePredictor:
    def __init__(self, batch_size=16, device=None, review_threshold=0.80):
        self.threshold = review_threshold

    def predict(self, text):
        confidence = 0.75 if "uncertain" in text else 0.95
        return apply_decision_policy({
            "label": "Positive",
            "confidence": confidence,
            "probabilities": {"Negative": 0.02, "Neutral": 0.98 - confidence, "Positive": confidence},
            "truncated": False,
        }, self.threshold)

    def predict_batch(self, texts):
        return [self.predict(text) for text in texts]


class CLITests(unittest.TestCase):
    def run_command(self, args, factory=FakePredictor):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                status = run_cli(factory, args)
            except SystemExit as error:
                status = error.code
        return status, stdout.getvalue(), stderr.getvalue()

    def test_single_prediction_returns_json_and_reports_time(self):
        status, stdout, stderr = self.run_command(["--text", "Profit increased."])
        self.assertEqual(status, 0)
        self.assertEqual(json.loads(stdout)["label"], "Positive")
        self.assertIn("Elapsed time", stderr)

    def test_default_threshold_sends_uncertain_prediction_to_review(self):
        status, stdout, _ = self.run_command(["--text", "uncertain"])
        self.assertEqual(status, 0)
        self.assertTrue(json.loads(stdout)["requires_review"])

    def test_batch_writes_ordered_predictions_and_review_flags(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "predictions.csv"
            source.write_text('text\n"Profit increased, significantly."\nuncertain\n', encoding="utf-8")
            status, _, stderr = self.run_command(["--input", str(source), "--output", str(output)])
            self.assertEqual(status, 0, stderr)
            with output.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual([row["text"] for row in rows], ["Profit increased, significantly.", "uncertain"])
            self.assertEqual([row["requires_review"] for row in rows], ["False", "True"])
            self.assertEqual(rows[1]["review_reasons"], "low_confidence")
            self.assertAlmostEqual(float(rows[0]["probability_positive"]), 0.95)
            self.assertIn("Elapsed time", stderr)

    def test_sentence_column_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "predictions.csv"
            source.write_text("sentence\nProfit increased.\n", encoding="utf-8")
            status, _, stderr = self.run_command(["--input", str(source), "--output", str(output)])
            self.assertEqual(status, 0, stderr)
            self.assertTrue(output.is_file())

    def test_invalid_input_is_rejected_before_loading_the_model(self):
        def forbidden_factory(**kwargs):
            self.fail("Invalid input should not load the model")

        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / "input.csv", Path(directory) / "predictions.csv"
            for content, expected in [(None, "not found"), ("wrong\nhello\n", "column"), ("text\n   \n", "empty"), ("text\n", "no data rows")]:
                with self.subTest(content=content):
                    if content is not None:
                        source.write_text(content, encoding="utf-8")
                    status, _, stderr = self.run_command(["--input", str(source), "--output", str(output)], forbidden_factory)
                    self.assertNotEqual(status, 0)
                    self.assertIn(expected, stderr.lower())
                    self.assertFalse(output.exists())

    def test_cannot_overwrite_input_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            source.write_text("text\nProfit increased.\n", encoding="utf-8")
            status, _, stderr = self.run_command(["--input", str(source), "--output", str(source)])
            self.assertNotEqual(status, 0)
            self.assertIn("same file", stderr)
            self.assertEqual(source.read_text(), "text\nProfit increased.\n")

    def test_argument_validation(self):
        for args in [["--text", "a", "--input", "b"], ["--input", "a"], ["--text", "a", "--output", "b"], ["--text", " "], ["--text", "a", "--batch-size", "0"]]:
            with self.subTest(args=args):
                status, _, stderr = self.run_command(args)
                self.assertNotEqual(status, 0)
                self.assertIn("error", stderr.lower())

    def test_help_describes_all_required_options(self):
        status, stdout, _ = self.run_command(["--help"])
        self.assertEqual(status, 0)
        for option in ["--text", "--input", "--output", "--batch-size", "--device"]:
            self.assertIn(option, stdout)

    def test_interactive_mode_predicts_and_exits(self):
        with patch("builtins.input", side_effect=["Profit increased.", "exit"]):
            status, stdout, stderr = self.run_command([])
        self.assertEqual(status, 0)
        self.assertIn('"label": "Positive"', stdout)
        self.assertIn("Elapsed time", stderr)

    def test_invalid_output_directory_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.csv"
            output = Path(directory) / "missing" / "output.csv"
            source.write_text("text\nProfit increased.\n", encoding="utf-8")
            status, _, stderr = self.run_command(["--input", str(source), "--output", str(output)])
            self.assertNotEqual(status, 0)
            self.assertIn("Invalid output file path", stderr)


if __name__ == "__main__":
    unittest.main()
