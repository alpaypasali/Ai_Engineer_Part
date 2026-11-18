# FinanceSentinel — Day 03: Sequence Length Experiments

**Date:** 2026-11-18  
**Task:** Controlled DistilBERT sequence-length analysis  
**Model:** `distilbert-base-uncased`  
**Dataset:** Financial PhraseBank, `sentences_allagree`, deduplicated  
**Primary metric:** Validation Macro-F1  
**Status:** Experiments complete; model selection provisional

## Executive Summary

This study compares `max_length=128` and `max_length=256` for three-class financial sentiment classification. Of 1,807 training examples, only one exceeds 128 tokens; none of the 226 validation examples exceeds 128 tokens. A 256-token limit yields a higher validation Macro-F1 (**0.958797 vs. 0.949958**) and Negative Recall (**0.966667 vs. 0.933333**) while training time is effectively unchanged in this run (**18.0875 s vs. 18.0439 s**). Peak PyTorch GPU memory allocation increases by **118.65 MB (+7.83%)**.

The models disagree on only **3/226 validation examples (1.33%)**, with a net gain of **one correct prediction** for the 256-token model. Since the validation token sequences are identical under both limits, these results **do not establish that observing longer validation inputs caused the improvement**. We provisionally advance the 256-token model for further evaluation, not production deployment.

## 1. Objective

Determine whether raising the maximum input sequence length from 128 to 256 improves financial sentiment classification enough to justify any additional training or GPU-memory cost. Assess the risk of truncation-induced information loss, compare error patterns, and make a documented model-selection decision without touching the held-out test set.

**Research questions:**

- What is the actual token-length distribution in the fixed train and validation splits?
- How much information would each configuration truncate?
- How do validation quality and training-resource metrics change when only `max_length` changes?
- Which validation examples produce different predictions, and what can these differences support causally?

## 2. Token Analysis

Token counts were computed with the DistilBERT tokenizer **without truncation or padding**, including `[CLS]` and `[SEP]`. Word count was based on whitespace splitting; `token_word_ratio = token_count / word_count`.

### Overall token-length statistics

| Statistic | Train (n=1,807) | Validation (n=226) |
|---|---:|---:|
| Minimum | 4 | 7 |
| Mean | 29.969 | 30.699 |
| Median | 27 | 28 |
| P90 | 49 | 50.5 |
| P95 | 57 | 57 |
| P99 | 69 | 64 |
| Maximum | 150 | 67 |

### Class-level token-length statistics

| Split | Class | Count | Mean | P95 | Maximum |
|---|---|---:|---:|---:|---:|
| Train | Negative | 242 | 33.504 | 61.00 | 75 |
| Train | Neutral | 1,109 | 27.507 | 51.00 | 150 |
| Train | Positive | 456 | 34.081 | 62.25 | 80 |
| Validation | Negative | 30 | 35.233 | 60.65 | 65 |
| Validation | Neutral | 139 | 26.935 | 50.10 | 62 |
| Validation | Positive | 57 | 37.491 | 62.40 | 67 |

Negative and Positive texts tend to be longer than Neutral texts in these samples. The training distribution is right-skewed and contains one long Neutral outlier.

**Statistical exploration:** A Kruskal–Wallis test on the train token lengths produced H = **102.6363**, p = **5.1620 × 10⁻²³**, rejecting the null of identical group distributions. Pairwise Mann–Whitney U tests with Holm correction yielded:

| Comparison | Holm-adjusted p-value | Rank-biserial effect | Significant at 0.05? |
|---|---:|---:|---|
| Negative vs. Neutral | 2.7598 × 10⁻¹¹ | +0.2769 | Yes |
| Negative vs. Positive | 0.7536 | −0.0144 | No |
| Neutral vs. Positive | 1.8575 × 10⁻¹⁸ | −0.2855 | Yes |

Statistical significance does **not** imply that sequence length alone is a sufficient predictor, or that the fine-tuned transformer relies on sequence length as a shortcut. In an auxiliary length-only Logistic Regression experiment, validation Macro-F1 was **0.468374**, versus **0.253881** for the majority-class baseline; this suggests a usable, but limited, length-related signal.

**Figure:** [Token length distribution](../figures/token_length_distribution.png)

## 3. Truncation Analysis

Truncation was analyzed using `lost_tokens = max(0, token_count - max_length)`. Average lost tokens are averaged across **all** examples in each split, including those losing zero tokens.

| Split | Max length | Truncated samples | Truncation rate | Total lost tokens | Mean lost tokens / sample |
|---|---:|---:|---:|---:|---:|
| Train | 128 | 1 | 0.05534% | 22 | 0.01218 |
| Train | 256 | 0 | 0% | 0 | 0 |
| Validation | 128 | 0 | 0% | 0 | 0 |
| Validation | 256 | 0 | 0% | 0 | 0 |

At 128 tokens, the only affected training example is **Neutral**, giving a class-specific truncation rate of **1/1,109 = 0.09017%**. Negative and Positive have zero truncated examples. At 256 tokens, every class has a zero truncation rate.

### Qualitative information-loss inspection

The longest training example contains **150 tokens**, and consists of supported Nokia phone-model names. At 128 tokens, the final 22 tokens are lost:

```text
##10 navigator, 6120 classic, 6110 navigator, 5700, 5500, 5320xm.
```

The truncation boundary splits a WordPiece representation of the product name `6210`, leaving `62` in the retained sequence and `##10` in the lost sequence. At 256 tokens the full message is retained. The removed material is a list of model identifiers rather than an obvious profit/loss statement, so a large semantic effect on its Neutral label is **not expected**; this is a qualitative judgment, not a directly measured counterfactual prediction.

The other four longest training examples contain **93, 80, 77, and 77 tokens** and are fully preserved by both limits. No synthetic examples were added to training.

## 4. Experimental Design

The controlled experiments shared the same preprocessed data, split membership, tokenizer family, architecture, evaluation procedure, and hyperparameters. **Only `max_length` differed.**

| Setting | Experiment A | Experiment B |
|---|---|---|
| Pretrained model | `distilbert-base-uncased` | Same |
| Maximum sequence length | **128** | **256** |
| Learning rate | 2e-5 | 2e-5 |
| Epochs | 3 | 3 |
| Train / evaluation batch size | 16 / 16 | 16 / 16 |
| Weight decay | 0.01 | 0.01 |
| Warmup optimizer steps | 34 | 34 |
| Random seed / data seed | 42 / 42 | 42 / 42 |
| Padding | Dynamic (`DataCollatorWithPadding`) | Dynamic |
| Learning-rate scheduler | Linear | Linear |
| Optimizer updates | 339 | 339 |
| Model selection | Best validation Macro-F1 | Best validation Macro-F1 |
| Best epoch | 3 | 3 |

Each run initialized a **fresh pretrained model and classification head**, Trainer, and optimizer. The 256-token run did not continue training the 128-token model. Checkpoints were stored in distinct output directories.

**Data integrity:** The fixed deduplicated split has 1,807 train, 226 validation, and 226 held-out test examples. All model-selection comparisons in this report use train/validation only; the held-out test set was not used for selecting sequence length.

## 5. Model Performance

All scores below are **validation** results from the controlled runs, not held-out test estimates.

| Metric | 128 tokens | 256 tokens | Difference (256 − 128) |
|---|---:|---:|---:|
| Accuracy | 0.964602 | **0.969027** | +0.004425 |
| Macro-F1 | 0.949958 | **0.958797** | +0.008839 |
| Macro Precision | 0.947096 | **0.950717** | +0.003621 |
| Macro Recall | 0.953040 | **0.967601** | +0.014561 |
| Negative Recall | 0.933333 | **0.966667** | +0.033333 |
| Neutral Recall | **0.978417** | 0.971223 | −0.007194 |
| Positive Recall | 0.947368 | **0.964912** | +0.017544 |
| Validation loss | **0.115270** | 0.117045 | +0.001775 |
| Mean training loss | 0.300558 | **0.297725** | −0.002833 |

The 256-token configuration scores higher in validation Macro-F1 and two class recalls, but the validation loss is slightly higher. This is plausible because cross-entropy responds to confidence as well as classification correctness, whereas F1 uses predicted labels.

On the 30 Negative validation examples, recall rises from **28/30** to **29/30**, a difference of only one correctly identified Negative example. Treat this improvement as preliminary.

**Figure:** [Sequence-length performance](../figures/sequence_length_performance.png)

## 6. Resource Usage

Both training runs used the same GPU setup and dynamic padding. Training time measured wall-clock time around `Trainer.train()`, synchronized on CUDA, including in-training evaluation and checkpoint operations. Peak memory was measured with `torch.cuda.max_memory_allocated()` and converted to **decimal MB** (`bytes / 1,000,000`); it excludes allocations outside the scope of PyTorch's allocator.

| Metric | 128 tokens | 256 tokens | Delta |
|---|---:|---:|---:|
| Training time | 18.043863 s | 18.087495 s | +0.043632 s (**+0.24%**) |
| Peak allocated GPU memory | 1,514.970624 MB | 1,633.617408 MB | +118.646784 MB (**+7.83%**) |
| Best epoch | 3 | 3 | 0 |

The very small timing gap is not sufficient to establish a true throughput difference from a single run. Peak memory is higher for 256 tokens, but not doubled: **dynamic padding** pads to the longest sequence in each batch rather than always to the configured maximum length.

**Figure:** [Sequence-length resource usage](../figures/sequence_length_resource_usage.png)

## 7. Disagreement Analysis

The prediction CSV was independently checked and the reported validation metrics were recomputed from its 226 rows.

| Prediction group | Examples |
|---|---:|
| Both models correct | **217** |
| Both models wrong | **6** |
| Only 128 correct | **1** |
| Only 256 correct | **2** |
| **Total** | **226** |

**Disagreement rate:** `3 / 226 = 1.3274%` (**1.33%**). The two models make identical predictions on **223/226 = 98.67%** of examples. The 128-token model gets **218** examples correct; the 256-token model gets **219** correct.

### Three disagreement cases

| True label | Token count | Pred. 128 | Pred. 256 | Relevant evidence |
|---|---:|---|---|---|
| Positive | 25 | Neutral | **Positive** | “sales and project resources will also be strengthened” |
| Neutral | 36 | **Neutral** | Positive | Net sales and operating profit figures without a comparative directional claim |
| Negative | 41 | Positive | **Negative** | Sales “slid” from EUR 91.2m to EUR 86.4m |

All three disagreement examples have token lengths **below 128**, so none is truncated in either configuration. In fact, every validation example has at most **67 tokens**. Therefore, given the same DistilBERT tokenizer and text, **the two max-length configurations yield identical validation `input_ids`**. Differences in predictions must come from differences in learned weights or execution details, not longer validation inputs.

**Causal caution:** Only one training example differs in retained content across the two configurations. This study cannot establish that the 22 extra training tokens caused the improvement; differences in model optimization, numerical execution, or other run-level variability may also contribute. The paired outcome changes are 2 wins for 256 and 1 win for 128; this sample is too small for a confident superiority claim.

**CSV:** [`prediction_disagreement.csv`](prediction_disagreement.csv)

## 8. Model Selection

**Decision: Provisionally select `max_length=256` for the next evaluation stage.**

**Reasons:**

- Higher validation Macro-F1 (**+0.008839**) and Negative Recall (**+0.033333**).
- Zero training truncation, compared with one truncated training example under 128.
- Practically equal observed training times for this single pair of runs.
- Additional peak allocated GPU memory of **118.65 MB**, an acceptable observed trade-off for this experiment, assuming deployment hardware can accommodate it.

**Decision confidence: Limited.** Two models disagree on only three validation examples, and the validation token inputs are the same. The result supports selecting the 256-token configuration as a *candidate*, not a claim that doubling maximum length generally improves performance.

**Next gates before deployment:** Repeat with several fixed seeds, measure inference latency and peak memory on deployment-like hardware, analyze errors by class and length bucket, then evaluate the frozen final choice **once** on the untouched test set.

## 9. Limitations

1. **Single-seed experiment:** One run per configuration cannot quantify run-to-run variability.
2. **Small validation set:** Only 226 examples, including 30 Negative examples; one corrected Negative case changes recall by 3.33 percentage points.
3. **Minimal actual length intervention:** Just one of 1,807 training examples exceeds 128 tokens; none of the validation examples does.
4. **No causal isolation of the source of the gain:** The 256-token run's better metrics cannot be directly attributed to additional meaningful financial information.
5. **Resource measurement scope:** Memory measures PyTorch allocated GPU memory rather than total device occupancy; timing can vary with GPU utilization, caching, and background processes.
6. **No inference benchmarking yet:** Training time and training peak memory do not establish production latency or inference memory.
7. **No final test performance:** The untouched test set must remain reserved until model-selection choices are finalized.
8. **Length associations are not causal:** Statistical tests and a length-only baseline establish association/predictive signal, not how the transformer makes decisions.

## 10. Tests — Ten Automated Checks

The script [`day_03_checks.py`](day_03_checks.py) was executed against the supplied **226-row** `prediction_disagreement.csv`. All ten checks below **passed**.

| # | Automated check | Result |
|---:|---|---|
| 1 | All expected CSV columns exist | **PASS** |
| 2 | Exactly 226 validation rows exist | **PASS** |
| 3 | No missing values in required columns | **PASS** |
| 4 | Actual labels and both predictions use class IDs 0, 1, 2 | **PASS** |
| 5 | Validation class counts match 30 Negative, 139 Neutral, 57 Positive | **PASS** |
| 6 | `same_prediction` matches both prediction columns | **PASS** |
| 7 | Both model-correctness flags match actual labels | **PASS** |
| 8 | Four disagreement groups and three differing predictions are consistent | **PASS** |
| 9 | Truncation flags and lost-token values match recorded token lengths | **PASS** |
| 10 | Recomputed Accuracy, Macro-F1, and Negative Recall reproduce prior results | **PASS** |

**Test outcome: `10/10 PASS`** (checks scoped to the uploaded **validation prediction CSV**). These checks do **not** rerun the model, regenerate tokenization, or independently validate the training notebook, split construction, benchmark figures, or GPU timings.

Run from the project root:

```bash
python reports/day_03_checks.py
```

## Reproducibility and Artifacts

Expected project artifacts:

```text
notebooks/03_sequence_length_experiments.ipynb
reports/token_length_statistics.csv
reports/truncation_analysis.csv
reports/sequence_length_comparison.csv
reports/prediction_disagreement.csv
reports/day_03_sequence_length_experiments.md
reports/day_03_checks.py
figures/token_length_distribution.png
figures/sequence_length_performance.png
figures/sequence_length_resource_usage.png
```

**Provenance note:** The prediction/disagreement counts and corresponding classification metrics were recomputed from the supplied CSV. Token-length, truncation, post-hoc, training-time, and GPU-memory results were transcribed from the completed notebook outputs shared during this project; raw training datasets, checkpoint weights, and resource logs were not supplied for independent reproduction here.
