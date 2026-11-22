from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def main():
    root = Path(__file__).resolve().parent
    model_path = root / "models" / "financesentinel_final"
    reports = root / "reports"
    required = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]

    for name in required:
        path = model_path / name
        assert path.is_file() and path.stat().st_size > 0, f"Missing or empty file: {name}"
        print(f"{name}: {path.stat().st_size / 1e6:.6f} MB")

    assert torch.cuda.is_available(), "CUDA is required to repeat Friday's GPU benchmark."
    device = torch.device("cuda:0")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, use_safetensors=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    parameter_count = sum(p.numel() for p in model.parameters())
    assert parameter_count == 66955779
    assert model.config.num_labels == 3
    assert {int(k): v.lower() for k, v in model.config.id2label.items()} == {
        0: "negative", 1: "neutral", 2: "positive"
    }
    assert next(model.parameters()).dtype == torch.float32
    print(f"Parameters: {parameter_count:,}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}; Transformers: {transformers.__version__}")

    df = pd.read_csv(root / "data" / "financial_sentiment.csv")
    df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
    train_df, temp_df = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
    )
    assert (len(train_df), len(val_df), len(test_df)) == (1807, 226, 226)
    previous_test = pd.read_csv(reports / "test_predictions.csv")
    assert np.array_equal(previous_test["row_id"].to_numpy(), test_df.index.to_numpy())
    assert np.array_equal(previous_test["actual"].to_numpy(), test_df["label"].to_numpy())
    texts = val_df["sentence"].tolist()

    def infer(batch):
        inputs = tokenizer(
            batch, padding=True, truncation=True, max_length=256, return_tensors="pt"
        ).to(device)
        with torch.inference_mode():
            probs = model(**inputs).logits.softmax(dim=1)
            confidence, predictions = probs.max(dim=1)
        return predictions.cpu(), confidence.cpu()

    torch.cuda.empty_cache()
    results = []
    reference = None

    for bs in [1, 8, 16, 32]:
        latencies, totals, peaks = [], [], []

        for _ in range(3):
            for _ in range(5):
                infer(texts[:bs])

            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            start_total = time.perf_counter()
            predictions = []

            for i in range(0, len(texts), bs):
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                pred, confidence = infer(texts[i:i + bs])
                torch.cuda.synchronize(device)
                latencies.append((time.perf_counter() - start) * 1000)
                predictions.append(pred)

            totals.append(time.perf_counter() - start_total)
            peaks.append(torch.cuda.max_memory_allocated(device) / 1e6)

            predictions = torch.cat(predictions)
            if reference is None:
                reference = predictions
            else:
                assert torch.equal(reference, predictions), f"Batch predictions differ: {bs}"

        results.append({
            "batch_size": bs,
            "mean_batch_latency_ms": np.mean(latencies),
            "p95_batch_latency_ms": np.percentile(latencies, 95),
            "throughput_samples_sec": len(texts) * 3 / sum(totals),
            "mean_total_time_sec": np.mean(totals),
            "peak_gpu_memory_mb": max(peaks)
        })

    benchmark = pd.DataFrame(results)
    assert np.isfinite(benchmark.to_numpy()).all()
    assert (benchmark.to_numpy() > 0).all()
    benchmark["device"] = torch.cuda.get_device_name(device)
    benchmark["torch_version"] = str(torch.__version__)
    benchmark["transformers_version"] = transformers.__version__
    benchmark.to_csv(reports / "inference_benchmark_clean.csv", index=False)
    print(benchmark.round(4).to_string(index=False))
    print("Clean-process model and benchmark checks: PASSED")


if __name__ == "__main__":
    main()
