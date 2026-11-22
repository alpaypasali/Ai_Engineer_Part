from pathlib import Path
import csv

import pandas as pd
from sklearn.model_selection import train_test_split

from src.decision_policy import analyze_thresholds, select_threshold
from src.predict import FinanceSentinel


def main():
    root = Path(__file__).resolve().parent
    data = pd.read_csv(root / "data" / "financial_sentiment.csv")
    data = data.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    train_df, temp_df = train_test_split(
        data, test_size=0.2, stratify=data["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    assert (len(train_df), len(val_df), len(test_df)) == (1807, 226, 226)
    assert val_df["label"].value_counts().to_dict() == {0: 30, 1: 139, 2: 57}

    reference_path = root / "reports" / "prediction_disagreement.csv"
    if reference_path.is_file():
        reference = pd.read_csv(reference_path)
        assert val_df["sentence"].tolist() == reference["sentence"].tolist()
        assert val_df["label"].tolist() == reference["actual_label"].tolist()

    predictor = FinanceSentinel(batch_size=16)
    predictions = predictor.predict_batch(val_df["sentence"].tolist())
    actual = [predictor.id2label[int(label)] for label in val_df["label"]]

    correct = sum(prediction["label"] == label for prediction, label in zip(predictions, actual))
    assert correct == 219, f"Validation accuracy changed: {correct}/{len(actual)}"
    if reference_path.is_file():
        assert [item["label"] for item in predictions] == [
            predictor.id2label[int(label)] for label in reference["prediction_256"]
        ]

    rows = analyze_thresholds(predictions, actual)
    report_path = root / "reports" / "threshold_analysis.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(pd.DataFrame(rows).to_string(index=False))
    selected = select_threshold(rows, target_accuracy=0.98)
    if selected is None:
        print("No threshold reached 98% accepted prediction accuracy.")
    else:
        print(f"\nRecommended threshold: {selected['threshold']:.2f}")
        print(f"Coverage: {selected['coverage']:.2%}")
        print(f"Accepted accuracy: {selected['accepted_prediction_accuracy']:.2%}")
        print(f"Rejected samples: {selected['rejected_samples']}")
    print(f"\nSaved: {report_path}")


if __name__ == "__main__":
    main()
