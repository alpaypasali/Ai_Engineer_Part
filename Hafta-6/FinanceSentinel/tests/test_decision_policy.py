import unittest

from src.decision_policy import apply_decision_policy, analyze_thresholds, select_threshold


class DecisionPolicyTests(unittest.TestCase):
    def test_default_policy_uses_selected_threshold(self):
        result = apply_decision_policy({"confidence": 0.85, "truncated": False})
        self.assertFalse(result["requires_review"])

    def test_normal_prediction_is_accepted_at_threshold(self):
        result = apply_decision_policy({"label": "Positive", "confidence": 0.90, "truncated": False}, 0.90)
        self.assertFalse(result["requires_review"])
        self.assertEqual(result["review_reasons"], [])
        self.assertEqual(result["label"], "Positive")

    def test_low_confidence_requires_review(self):
        result = apply_decision_policy({"confidence": 0.89, "truncated": False}, 0.90)
        self.assertTrue(result["requires_review"])
        self.assertEqual(result["review_reasons"], ["low_confidence"])

    def test_truncated_input_requires_review_even_at_high_confidence(self):
        result = apply_decision_policy({"confidence": 0.99, "truncated": True}, 0.90)
        self.assertTrue(result["requires_review"])
        self.assertIn("truncated_input", result["review_reasons"])

    def test_both_review_reasons_are_preserved(self):
        result = apply_decision_policy({"confidence": 0.50, "truncated": True}, 0.90)
        self.assertEqual(set(result["review_reasons"]), {"low_confidence", "truncated_input"})

    def test_invalid_confidence_and_threshold(self):
        for value in [float("nan"), -0.1, 1.1]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    apply_decision_policy({"confidence": value, "truncated": False}, 0.90)
                with self.assertRaises(ValueError):
                    apply_decision_policy({"confidence": 0.90, "truncated": False}, value)
        with self.assertRaises(TypeError):
            apply_decision_policy({"confidence": 0.90, "truncated": False}, "0.90")

    def test_threshold_analysis_excludes_truncated_predictions(self):
        predictions = [
            {"label": "Positive", "confidence": 0.95, "truncated": False},
            {"label": "Negative", "confidence": 0.80, "truncated": False},
            {"label": "Neutral", "confidence": 0.99, "truncated": True},
            {"label": "Positive", "confidence": 0.60, "truncated": False},
        ]
        rows = analyze_thresholds(predictions, ["Positive", "Positive", "Neutral", "Negative"], [0.80, 0.90, 0.99])
        self.assertEqual(rows[0], {"threshold": 0.80, "coverage": 0.50, "accepted_prediction_accuracy": 0.50, "rejected_samples": 2})
        self.assertEqual(rows[1], {"threshold": 0.90, "coverage": 0.25, "accepted_prediction_accuracy": 1.0, "rejected_samples": 3})
        self.assertIsNone(rows[2]["accepted_prediction_accuracy"])
        self.assertEqual(select_threshold(rows)["threshold"], 0.90)

    def test_selection_maximizes_coverage_at_accuracy_target(self):
        rows = [
            {"threshold": 0.70, "coverage": 0.98, "accepted_prediction_accuracy": 0.97},
            {"threshold": 0.80, "coverage": 0.94, "accepted_prediction_accuracy": 0.98},
            {"threshold": 0.90, "coverage": 0.85, "accepted_prediction_accuracy": 0.99},
        ]
        self.assertEqual(select_threshold(rows)["threshold"], 0.80)
        self.assertIsNone(select_threshold(rows, target_accuracy=1.0))

    def test_mismatched_or_empty_validation_data(self):
        with self.assertRaises(ValueError):
            analyze_thresholds([], [])
        with self.assertRaises(ValueError):
            analyze_thresholds([{"confidence": 0.95, "truncated": False}], [])


if __name__ == "__main__":
    unittest.main()
