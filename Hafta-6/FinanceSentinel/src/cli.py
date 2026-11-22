import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter


def build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="FinanceSentinel: financial sentiment classification.",
        epilog="Run without arguments for interactive mode. Input CSV must contain a text or sentence column.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--text", help="Classify a single financial sentence as JSON")
    mode.add_argument("--input", type=Path, help="Input CSV file for batch prediction")
    parser.add_argument("--output", type=Path, help="Output CSV file (required with --input)")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--device", choices=["cpu", "cuda"], help="Device (default: auto-detect)")
    return parser


def read_texts(path):
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, strict=True)
        column = next((name for name in ("text", "sentence") if name in (reader.fieldnames or [])), None)
        if column is None:
            raise ValueError("Input CSV must contain a 'text' or 'sentence' column")
        texts = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Invalid CSV row at line {reader.line_num}")
            if not row[column].strip():
                raise ValueError(f"Empty text at CSV line {reader.line_num}")
            texts.append(row[column])
    if not texts:
        raise ValueError("Input CSV has no data rows")
    return texts


def write_predictions(path, texts, predictions):
    rows = []
    for text, prediction in zip(texts, predictions, strict=True):
        probabilities = {label.lower(): score for label, score in prediction["probabilities"].items()}
        rows.append({
            "text": text,
            "label": prediction["label"],
            "confidence": prediction["confidence"],
            "probability_negative": probabilities["negative"],
            "probability_neutral": probabilities["neutral"],
            "probability_positive": probabilities["positive"],
            "truncated": prediction["truncated"],
            "requires_review": prediction["requires_review"],
            "review_reasons": ";".join(prediction["review_reasons"]),
        })
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_cli(predictor_factory, argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if (args.input is None) != (args.output is None):
        parser.error("--input and --output must be used together")
    if args.text is not None and not args.text.strip():
        parser.error("--text cannot be empty")

    start = perf_counter()
    try:
        if args.input is not None:
            texts = read_texts(args.input)
            if args.input.resolve() == args.output.resolve() or (
                args.output.exists() and args.input.samefile(args.output)
            ):
                raise ValueError("Input and output cannot be the same file")
            if not args.output.parent.is_dir() or args.output.is_dir():
                raise ValueError(f"Invalid output file path: {args.output}")

        predictor = predictor_factory(batch_size=args.batch_size, device=args.device)

        if args.text is not None:
            print(json.dumps(predictor.predict(args.text), indent=4, ensure_ascii=False))
        elif args.input is not None:
            predictions = predictor.predict_batch(texts)
            write_predictions(args.output, texts, predictions)
            print(f"Saved {len(predictions)} predictions to {args.output}")
            print(f"Manual review: {sum(item['requires_review'] for item in predictions)}")
        else:
            return interactive_mode(predictor)
    except (OSError, ValueError, TypeError, RuntimeError, csv.Error, ImportError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Elapsed time: {perf_counter() - start:.4f} seconds", file=sys.stderr)
    return 0


def interactive_mode(predictor):
    print("\nFinanceSentinel is ready. Type 'exit' to quit.\n")
    while True:
        try:
            text = input("Enter sentence: ").strip()
            if text.lower() == "exit":
                return 0
            start = perf_counter()
            print(json.dumps(predictor.predict(text), indent=4, ensure_ascii=False))
            print(f"Elapsed time: {perf_counter() - start:.4f} seconds", file=sys.stderr)
        except (ValueError, TypeError) as error:
            print(f"Error: {error}", file=sys.stderr)
        except (EOFError, KeyboardInterrupt):
            return 0
        print()
