# Day 04 — Hugging Face Classification Evaluation

## Problem

In ReviewLens, I evaluated how well a pretrained text-classification model predicts Yelp review ratings from 1 to 5 stars. The goal was to measure performance, identify the hardest rating classes, and examine typical mistakes without training new model weights.

## Dataset

- **Dataset:** Yelp Review Full
- **Local sample:** `data/yelp_sample_1000.csv` (1,000 reviews, 200 for each star rating)
- **Columns:** `text`, `label` (0–4), and `stars` (1–5)
- **Source split:** The local sample comes from the original Yelp *train* split, not the original *test* split.

## Evaluation Data

I selected **200 reviews**, with **40 per star rating**, using `train_test_split(test_size=0.20, stratify=df["stars"], random_state=42)`. The selected original row indices were saved in `data/eval_indices.csv` so the same development-evaluation set can be reused when comparing models on Friday.

These are **development evaluation** results, **not** independent final-test results. Some examples may also have appeared in the earlier pilot experiment.

## Model

- **Model ID:** `nlptown/bert-base-multilingual-uncased-sentiment`
- **Architecture:** Pretrained multilingual BERT for 1–5 star sentiment classification
- **Training in this experiment:** None; weights were left unchanged.

## Pipeline

I used Hugging Face `pipeline("text-classification", model=MODEL_NAME)` to run inference. The pipeline returns a predicted label and a score for each review.

## Label Mapping

The model returns labels as strings, while Yelp's `stars` column contains integers. I mapped the model labels to the same 1–5 representation used by the reference labels.

| Model class ID | Pipeline label | Comparable Yelp stars |
|---:|---|---:|
| 0 | `1 star` | 1 |
| 1 | `2 stars` | 2 |
| 2 | `3 stars` | 3 |
| 3 | `4 stars` | 4 |
| 4 | `5 stars` | 5 |

A dictionary-based conversion function rejects invalid labels such as `6 stars` with `ValueError`.

## Inference

- **Evaluation reviews:** 200
- **Batch size:** 8
- **Truncation:** `True`
- **Maximum input length:** 256 tokens, including special tokens
- **Output:** 200 labels and 200 corresponding scores

`y_true` and `y_pred` both have shape `(200,)`. The evaluation order was preserved when making predictions.

## Accuracy

**Accuracy = 0.5550** (111 correct and 89 incorrect out of 200).

## Macro-F1

The primary metric is **Macro-F1 = 0.553163**. Each star class contributes equally to this metric.

| Metric | Value |
|---|---:|
| Evaluation samples | 200 |
| Accuracy | 0.555000 |
| Macro Precision | 0.560473 |
| Macro Recall | 0.555000 |
| **Macro-F1** | **0.553163** |

## Per-Class Recall

| Actual stars | Samples | Correct | Recall | F1 (rounded) |
|---:|---:|---:|---:|---:|
| 1 | 40 | 24 | 0.600 | 0.67 |
| 2 | 40 | 22 | 0.550 | 0.54 |
| 3 | 40 | 15 | 0.375 | 0.42 |
| 4 | 40 | 20 | 0.500 | 0.47 |
| 5 | 40 | 30 | 0.750 | 0.67 |

The model did best on the **1-star** and **5-star** categories by F1, and it had the lowest recall on **3-star** reviews. This suggests that mixed or moderate reviews may be harder to distinguish, but review text would need to be examined to investigate that explanation.

## 1-Star Error Rate

The 1-star error rate counts **all incorrect predictions** for reviews whose actual rating is 1 star, regardless of how many stars the prediction differs by.

`1-Star Error Rate = Incorrect predictions for actual 1-star reviews / All actual 1-star reviews`

`= 16 / 40 = 0.40 = 40%`

This matches `1 - Recall(1-star) = 1 - 0.60 = 0.40`.

## Confusion Matrix

**Rows are actual stars; columns are predicted stars.**

| Actual \ Predicted | 1 | 2 | 3 | 4 | 5 |
|---|---:|---:|---:|---:|---:|
| **1** | 24 | 14 | 1 | 1 | 0 |
| **2** | 6 | 22 | 10 | 1 | 1 |
| **3** | 2 | 4 | 15 | 14 | 5 |
| **4** | 0 | 1 | 6 | 20 | 13 |
| **5** | 0 | 0 | 0 | 10 | 30 |

The entries sum to **200**, with **111** on the diagonal and **89** off it. **77 of 89 errors** were between adjacent star categories, approximately **86.5%** of all errors. The biggest one-way confusions were actual **1 → predicted 2** and **3 → 4**, both with 14 cases.

Visualization saved locally as `figures/hf_confusion_matrix.png`. The graph should be checked for retained raw review text before publishing; the image itself contains aggregate counts only.

## Error Analysis

I selected five incorrect predictions from different actual-star groups. These are all **one-star-off** errors.

| Row ID | Actual stars | Predicted stars | Model score |
|---:|---:|---:|---:|
| 638 | 1 | 2 | 0.339196 |
| 599 | 1 | 2 | 0.496093 |
| 907 | 2 | 1 | 0.476728 |
| 673 | 3 | 2 | 0.633028 |
| 776 | 4 | 5 | 0.551618 |

The highest score in these five selected mistakes was about **0.633** (row 673). These are model confidence-like scores, not guarantees of correctness. I did not include review text in the report. Without checking the full text, I cannot claim that mixed sentiment, truncation, or rating ambiguity definitively caused an individual mistake.

## Limitations

1. The evaluation set is a small, class-balanced sample (200 rows) from Yelp's original training split; it is not an independent held-out Yelp test benchmark.
2. This model was pretrained for product reviews, so performance may differ on restaurant reviews.
3. `max_length=256` can truncate long reviews. Important negative or positive statements near the end might be lost.
4. The pipeline score is not necessarily a calibrated probability of being correct.
5. A 1-star prediction error of 1 → 2 and one of 1 → 5 are counted equally by the 1-star error-rate metric, though their practical impact may differ.
6. The selected five errors are illustrative, not a representative statistical analysis of every error pattern.

## Tests

Three automated tests were run in the notebook, and the user reported that each passed.

| Test | Result |
|---|---|
| Label Mapping (including invalid labels) | PASSED |
| Prediction Alignment (200 samples, valid labels, matching order) | PASSED |
| Evaluation Metrics (confusion matrix and 1-star error identity) | PASSED |

## What I Learned

I learned how to use a pretrained Hugging Face classification model without updating its weights. I also learned why I have to map labels carefully before evaluating predictions. Accuracy gave me an overall result, but per-class recall and Macro-F1 showed that the model had more difficulty with middle ratings, especially 3 stars. Confusion matrix made the mistakes easier to understand: most predictions were off by only one star. Finally, I learned that a high pipeline score does not always mean a correct prediction.

## Reproducibility and Prediction Export

For Friday's comparison, reuse the saved `data/eval_indices.csv` and the same `max_length=256` setting. The full **200-row prediction CSV must be exported from the original notebook session**, since only five verified predictions were shared in the conversation. A notebook-ready export cell is included in `export_full_hf_predictions.py` and will replace the partial file with all 200 verified predictions when run in the notebook namespace.

## Data-Sharing Note

No raw review text is included in this report or the example prediction CSV. Do **not** publish Yelp raw reviews, notebooks retaining raw review outputs, or derived artifacts without confirming the dataset license/terms permit redistribution. Keep `hf_evaluation_predictions.csv` private until that verification is complete. The numerical row IDs refer to the local working sample and are not global Yelp IDs.
