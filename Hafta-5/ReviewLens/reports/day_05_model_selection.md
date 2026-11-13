# Day 05 — Model Selection: ReviewLens

**Project:** ReviewLens · **Date:** 2026-11-13 · **Notebook:** `notebooks/06_model_selection.ipynb`

## 1. Problem

The goal is to classify Yelp customer reviews into **1–5 stars**, with particular attention to detecting genuinely negative (**1-star**) experiences. This comparison considers predictive performance, the cost of missing low-rated reviews, inference latency, output compatibility, and licensing. We did **not** fine-tune either candidate.

## 2. Candidate Models

| | Model A | Model B |
|---|---|---|
| HF model ID | `nlptown/bert-base-multilingual-uncased-sentiment` | `tabularisai/multilingual-sentiment-analysis` |
| Architecture | Multilingual BERT (uncased) | Multilingual DistilBERT (cased) |
| Primary training task | 1–5 star sentiment on **product reviews** | Five-level sentiment classification |
| License (as listed in model card) | MIT | CC BY-NC 4.0 |

Sources: [Model A card](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment) · [Model B card](https://huggingface.co/tabularisai/multilingual-sentiment-analysis).

## 3. Evaluation Data

- Dataset: Yelp Review Full; local sample `data/yelp_sample_1000.csv` (1,000 reviews).
- Evaluation: **200 reviews**, balanced at **40 reviews per star rating**.
- Both models used **identical review row IDs and actual labels in identical order**, confirmed in the notebook's alignment checks.
- Evaluation row IDs come from `data/eval_indices.csv`; **no new train/test split** was created on Day 05.
- These are development-evaluation results, not proof of performance on a fully independent production/test set.

## 4. Label Mapping

**Model A:** `1 star → 1`, `2 stars → 2`, `3 stars → 3`, `4 stars → 4`, `5 stars → 5`. The output directly models star ratings.

**Model B:** `Very Negative → 1`, `Negative → 2`, `Neutral → 3`, `Positive → 4`, `Very Positive → 5`.

**Caveat:** Model B's sentiment-to-star mapping is an evaluation **assumption**, not a mathematical equivalence. A text with 'Negative' sentiment may still have an actual Yelp rating of 1 star. This mapping can contribute to apparent classification errors.

## 5. Performance

Both candidates were evaluated from their **saved 200-row prediction CSVs**; Model A's values were checked against the prior day's report.

| Metric | Model A | Model B |
|---|---:|---:|
| Evaluation samples | 200 | 200 |
| Accuracy | **0.5550** | 0.4200 |
| Macro Precision | **0.5605** | 0.4319 |
| Macro Recall | **0.5550** | 0.4200 |
| Macro-F1 | **0.5532** | 0.3827 |
| 1-Star Recall | **0.6000** | 0.5000 |
| 1-Star Error Rate | **0.4000** | 0.5000 |

Model A has the stronger five-class performance. Model B's poorer Macro-F1 is a combination of task/domain fit and the limitations of the approximate label mapping; the current evaluation cannot isolate those causes.

## 6. One-Star Error

Formula: **1-Star Error Rate = wrong predictions among actual 1-star reviews / number of actual 1-star reviews = 1 − Recall of 1-star class**.

- **Model A:** 16 wrong / 40 actual 1-star reviews = **0.40**. Recall = 0.60.
- **Model B:** 20 wrong / 40 actual 1-star reviews = **0.50**. Recall = 0.50.

Compared with Model B, Model A correctly identifies **four additional 1-star reviews** in the current 40-review subgroup. Because low-star experiences are important to ReviewLens, this is a meaningful consideration.

## 7. Inference Latency

Latency was measured for **the same 20 reviews** (four per true-star class), in the same order, on **`cuda:0`**. Both models were loaded and warmed up before measuring. Sequential single-text inference used `batch_size=1`, `truncation=True`, and `max_length=256`; GPU operations were synchronized around timing. Model download/loading time was excluded.

| Measure | Model A | Model B |
|---|---:|---:|
| Measured reviews | 20 | 20 |
| Mean latency | **16.7591 ms** | **7.1606 ms** |
| Median latency | 11.3798 ms | **5.3383 ms** |
| P95 latency | 32.2583 ms | **14.6093 ms** |

Model B is approximately **2.34× faster** by mean latency. These are exploratory measurements on 20 inputs; they do not establish production latency or reliable production P95. The screenshot-based numeric values have limited printed precision. GPU scheduling, padding, input lengths, caching, and other execution conditions can affect timing. Throughput was **not measured** here.

> **Latency CSV integrity:** The available screenshots preserve all **40 measured latency values** but not the corresponding 20 `row_id`s. Therefore, `reports/inference_latency.csv` contains 40 measured values with **blank `row_id` fields**. Run `reports/finalize_inference_latency.py` inside the original notebook to fill the true IDs from `latency_eval` and verify sample alignment. Do not treat this CSV as finalized until all row IDs are populated.

## 8. License Comparison

- **Model A — MIT:** Generally permits commercial use if license conditions are respected.
- **Model B — CC BY-NC 4.0:** Non-commercial license; commercial deployment may require separate authorization from the rights holder.
- **Dataset rights are separate:** MIT licensing of Model A does **not** automatically permit redistribution or commercial exploitation of raw Yelp content. The Yelp dataset terms must be checked independently.

License metadata is based on the Hugging Face model cards at the time of writing. This is not legal advice.

## 9. Model Limitations & Risks

**Model A:** Trained on product-review sentiment rather than specifically adapted to Yelp; struggles with ambiguous/middle-star cases (especially 3-star reviews); long reviews may be truncated at 256 tokens; its 1-star error rate is still 40%. A falsely high rating can hide negative customer experiences.

**Model B:** Predicts sentiment levels rather than actual Yelp stars; the mapping introduces label uncertainty; lower observed overall and 1-star recall; non-commercial license; longer reviews may be truncated; classification scores should not be interpreted as calibrated probabilities without separate validation.

**Shared issues:** The two tokenizers differ, so `max_length=256` does not guarantee retaining exactly the same character span. The 200 development samples may not represent production traffic. Repeated benchmarks and an independent holdout are needed before deployment.

## 10. Model Selection

**Selected candidate: Model A — `nlptown/bert-base-multilingual-uncased-sentiment`.**

Model A offers higher **Macro-F1** (0.5532 vs 0.3827) and lower **1-Star Error Rate** (40% vs 50%). It directly predicts star ratings, without an extra sentiment-to-rating assumption, and its **MIT** license is more compatible with potential commercial use than Model B's CC BY-NC 4.0. Model B has lower latency (7.16 ms vs 16.76 ms), but the measured difference may be acceptable if the application prioritizes accurate identification of poor experiences. An explicit product latency budget is still needed.

This is a **candidate selection**, **not a production deployment approval**. Further independent testing, error analysis, business thresholds, data licensing review, and operational checks remain necessary.

## 11. Tests

Three automated notebook tests were **written**:

1. **Label Mapping** — both five-class mappings and invalid Model B label rejection.
2. **Evaluation Alignment** — 200 predictions per model, matching `row_id` and real stars, valid 1–5 predictions, 40 examples per class.
3. **Metric Consistency** — 5×5 confusion matrices summing to 200 and `1-Star Error Rate = 1 − 1-Star Recall` for both models.

**Execution status:** User has not yet shared the three tests' final PASS outputs. The report does not claim they passed; rerun the notebook's tests before marking them PASSED.

## 12. What I Learned

I learned that choosing a model is not the same as choosing the highest speed or one isolated metric. In ReviewLens, Model B was faster but Model A better classified the five star ratings, especially genuine 1-star feedback. The model's original output labels matter: treating sentiment levels as exact star ratings introduces a meaningful assumption. I also learned to compare the **same samples**, verify **row alignment**, measure latency on the **same device** after warm-up, and examine **commercial licensing** separately from model quality and dataset rights.

## 13. Artifacts

- `reports/day_05_model_selection.md` — this report.
- `reports/model_comparison.csv` — two-model summary.
- `reports/inference_latency.csv` — 40 measured latencies; `row_id`s require finalization from the notebook.
- `reports/finalize_inference_latency.py` — notebook exporter that fills and validates actual row IDs.
- `notebooks/06_model_selection.ipynb` — source of the measurements and the three automated tests.

**Privacy/licensing:** The report and comparison CSV contain aggregate information, not raw Yelp text. Before publishing row-level data (including IDs and derived outputs) to a public GitHub repository, verify that redistribution is permitted by the data source's terms.
