# HealthBrief — Day 01 Summarization Baseline

## 1. Project Overview

HealthBrief investigates biomedical research summarization using extractive and abstractive approaches.

The objective is to evaluate summarization quality, factual consistency, computational performance, and limitations before fine-tuning.

## 2. Dataset Selection

Dataset: BioLaySumm/BioLaySumm2025-eLife

Dataset revision: 144e9c785e3a309da804eb6786feb38e6596f390

Sampling seed: 42

Dataset splits:

- Train: 500
- Validation: 100
- Test: 100

The original dataset splits were preserved.

The test split does not contain reference summaries. Therefore, reference-based metrics cannot be calculated on this split.

The dataset does not provide an explicit license declaration in its Hugging Face metadata. Commercial use and redistribution permissions have not been established.

## 3. Data Quality

Data quality checks included:

- Missing values
- Empty text detection
- Duplicate articles
- Duplicate titles
- Cross-split overlap

Automated test results are provided in Section 8.

## 4. Length Analysis

| split | column | mean_word_count | p50_token_count | p95_token_count | p99_token_count |
| --- | --- | --- | --- | --- | --- |
| train | article | 10008.12 | 16272.0 | 27737.45 | 33191.25 |
| train | summary | 381.72 | 519.0 | 688.15 | 760.05 |
| validation | article | 9847.24 | 15754.0 | 27568.25 | 30393.81 |
| validation | summary | 387.93 | 508.5 | 691.1 | 744.15 |

Biomedical research articles are substantially longer than the input lengths commonly used in standard summarization baselines.

Train articles contain approximately 10,008 words on average.

Reference summaries contain approximately 382 words on average.

The average article-to-summary compression ratio is approximately 27.02 for training data and 26.23 for validation data.

### Truncation Analysis

| split | max_input_length | mean_original_tokens | mean_retained_tokens | mean_lost_tokens | token_loss_rate |
| --- | --- | --- | --- | --- | --- |
| train | 256 | 17033.36 | 256.0 | 16777.36 | 98.5 |
| train | 512 | 17033.36 | 512.0 | 16521.36 | 96.99 |
| train | 1024 | 17033.36 | 1024.0 | 16009.36 | 93.99 |
| validation | 256 | 16729.98 | 256.0 | 16473.98 | 98.47 |
| validation | 512 | 16729.98 | 512.0 | 16217.98 | 96.94 |
| validation | 1024 | 16729.98 | 1024.0 | 15705.98 | 93.88 |

All sampled articles exceed 1024 tokens.

At a 512-token input limit, approximately 97% of source tokens are discarded.

Token loss does not necessarily equal semantic information loss.

## 5. Baseline Models

Two summarization approaches were evaluated on the same 20 validation articles.

### Model A — Lead-3

Extractive summarization using the first three sentences of each article.

### Model B — FLAN-T5-small

Zero-shot abstractive summarization.

Configuration:

- Model: google/flan-t5-small
- Maximum input length: 512
- Maximum new tokens: 512
- Greedy decoding
- No fine-tuning

### Baseline Results

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L | Inference Time (s) |
| --- | --- | --- | --- | --- |
| Lead-3 | 17.61 | 4.72 | 10.52 | 0.0004 |
| FLAN-T5-small | 4.33 | 1.14 | 3.47 | 0.3837 |

ROUGE values represent average F1 scores multiplied by 100.

Lead-3 achieved higher ROUGE scores than zero-shot FLAN-T5-small.

These results do not establish that Lead-3 produces more factually reliable summaries.

Inference time measurements are preliminary and do not represent a fully standardized hardware benchmark.

## 6. Factual Consistency and Hallucination

Five FLAN-T5-small summaries were manually reviewed.

Reviewed sample IDs: 7, 8, 12, 17, 36.

### Observed Issues

- Sample 7: The summary omits the main findings concerning FLC gene expression and epigenetic memory.
- Sample 8: The generated terminology refers to coiled-coil membranes rather than the reported middle domains.
- Sample 12: The model incorrectly states that TBP2 is required for muscle-gene expression, contradicting the study findings.
- Sample 17: The summary omits the findings involving the NALCN ion channel.
- Sample 36: The model emphasizes forensic entomology instead of the study's microbial method for estimating postmortem intervals.

### Error Category Counts

- unsupported_information: 1
- incorrect_numeric: 0
- exaggerated_result: 0
- important_omission: 5
- uncertainty_to_certainty: 1

The review used selected source excerpts and reference summaries. A complete claim-level audit against the full articles is still required.

The findings from five examples must not be interpreted as the model's overall hallucination rate.

## 7. Model Limitations

### Long-Document Truncation

The model receives only a small portion of each biomedical article.

Important findings may appear outside the retained input.

### Short Generated Summaries

FLAN-T5-small generated very short summaries compared with the reference summaries.

### Limited Model Capacity

A small general-purpose instruction-tuned model may struggle with specialized biomedical summarization.

### Factual Consistency

ROUGE measures lexical overlap rather than factual correctness.

Negation, uncertainty, scientific terminology, and research findings require separate verification.

### Evaluation Size

The baseline evaluation contains only 20 validation examples.

The manual factual consistency analysis covers five generated summaries.

### Licensing

The dataset's licensing information requires further clarification before redistribution or commercial use.

## 8. Automated Testing

| Test | Status |
| --- | --- |
| Split Integrity | PASS |
| Data Leakage | PASS |
| Valid Text | PASS |
| Input Length | PASS |
| Output Validity | FAIL |
| Metric Validity | PASS |

Tests cover split integrity, exact normalized cross-split article overlap, required text validity, truncated input length, output validity, and metric range validity.

## 9. Conclusion

Lead-3 outperformed FLAN-T5-small in the initial ROUGE comparison.

However, ROUGE performance alone cannot establish factual reliability.

The most significant technical challenge is the large difference between article length and available model input context.

Future experiments should investigate approaches that preserve more source information while evaluating factual consistency.

No fine-tuning was performed during this baseline experiment.
