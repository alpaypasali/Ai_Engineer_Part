# FinanceSentinel — Data Card

**Project:** FinanceSentinel  
**Date:** 2026-11-16  
**Notebook:** `notebooks/01_data_analysis.ipynb`

## 1. Problem

FinanceSentinel aims to classify English financial news sentences into three sentiment categories: Negative, Neutral, and Positive. The goal is to identify the financial meaning of a sentence from an investor's perspective, rather than predict future stock prices.

The main evaluation metric will be Macro-F1, supported by per-class Recall, because the dataset is imbalanced.

## 2. Dataset

| Property | Description |
|---|---|
| Dataset | Financial PhraseBank |
| Source | [takala/financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank) |
| Language | English |
| Task | Multiclass financial sentiment classification |
| Features | `sentence`, `label` |
| Original split | Train only; no separate validation or test set |
| Loading revision | `refs/pr/10` (Parquet conversion in the original repository) |
| License | CC BY-NC-SA 3.0 |

The corpus contains financial news sentences annotated by people with financial knowledge. Each sentence was assessed by five to eight annotators. The four configurations represent different levels of agreement.

### License and permitted use

The dataset card specifies [Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported](https://creativecommons.org/licenses/by-nc-sa/3.0/).

- **Attribution:** Credit the authors and source, link the license, and identify adaptations when applicable.
- **NonCommercial:** Commercial use requires separate permission from the rights holders.
- **ShareAlike:** Distribute adaptations of the licensed material under the same or a compatible license.

The project is intended for non-commercial educational use. The dataset license is separate from the license of any pretrained model or project code. Before distributing the raw data, processed data, or trained model, verify the applicable rights and conditions, including any additional rights in the source news text. Keep the raw CSV local by default.

## 3. Label Mapping

The mapping was verified using the dataset's `ClassLabel` metadata in all four configurations.

| Label ID | Sentiment |
|---:|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

## 4. Agreement Analysis

### Dataset size and class distribution

| Configuration | Minimum agreement | Total | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|---:|
| 50agree | 50% | 4,846 | 604 | 2,879 | 1,363 |
| 66agree | 66% | 4,217 | 514 | 2,535 | 1,168 |
| 75agree | 75% | 3,453 | 420 | 2,146 | 887 |
| allagree | 100% | 2,264 | 303 | 1,391 | 570 |

Neutral is the majority class and Negative is the minority class in every configuration. Always predicting Neutral in `50agree` would achieve approximately 59.4% accuracy while failing to identify every Negative and Positive example. Therefore, accuracy alone is insufficient.

Related artifacts:

- `reports/agreement_comparison.csv`
- `figures/class_distribution.png`

### Unique sentence intersection

| Configuration | 50agree | 66agree | 75agree | allagree |
|---|---:|---:|---:|---:|
| 50agree | 4,838 | 4,211 | 3,448 | 2,259 |
| 66agree | 4,211 | 4,211 | 3,448 | 2,259 |
| 75agree | 3,448 | 3,448 | 3,448 | 2,259 |
| allagree | 2,259 | 2,259 | 2,259 | 2,259 |

All 2,259 unique sentences in `allagree` are also present in `50agree`. No label mismatch was found among those common sentences.

There are 2,579 unique sentences in `50agree` that do not appear in `allagree`.

| Non-overlapping agreement group | Unique sentences |
|---|---:|
| 50agree only (not in 66agree) | 627 |
| 66agree, excluding 75agree | 763 |
| 75agree, excluding allagree | 1,189 |
| allagree | 2,259 |

Five unique examples were sampled from each group for manual inspection (20 examples in total).

Higher agreement provides stronger annotation consensus, not a guarantee of correctness. Using one configuration as training data and a contained configuration as test data would cause data leakage. Concatenating the four configurations would not add unique sentences beyond `50agree` and would duplicate many examples.

## 5. Data Quality

All counts use the original configurations before deduplication.

| Configuration | Missing text | Missing labels | Empty text | Duplicate sentences | Conflicting sentences |
|---|---:|---:|---:|---:|---:|
| 50agree | 0 | 0 | 0 | 8 | 2 |
| 66agree | 0 | 0 | 0 | 6 | 0 |
| 75agree | 0 | 0 | 0 | 5 | 0 |
| allagree | 0 | 0 | 0 | 5 | 0 |

Duplicate counts represent repeated sentence occurrences beyond the first occurrence. Conflicting counts represent unique sentences associated with more than one label.

Conflicting labels can provide contradictory learning signals and make the optimization objective harder to satisfy.

### Sentence length (whitespace-separated words)

| Configuration | Minimum | Mean | Maximum |
|---|---:|---:|---:|
| 50agree | 2 | 23.10 | 81 |
| 66agree | 2 | 23.02 | 81 |
| 75agree | 2 | 22.76 | 81 |
| allagree | 2 | 22.44 | 81 |

The distributions are right-skewed, with most sentences approximately 10–30 words long. Word count is not the same as tokenizer sequence length; token lengths should be measured before selecting a maximum input length.

Related artifacts:

- `reports/data_quality.csv`
- `figures/sentence_length_distribution.png`

## 6. Selected Dataset

**Selected configuration: `sentences_allagree`.**

This configuration contains 2,264 rows and has full annotator agreement. It has no missing sentences, missing labels, empty sentences, or conflicting labels. Five repeated sentence occurrences were identified.

| Class | Original count | Share |
|---|---:|---:|
| Negative | 303 | 13.38% |
| Neutral | 1,391 | 61.44% |
| Positive | 570 | 25.18% |

The selection prioritizes annotation consistency for the first fine-tuning experiment, accepting a smaller dataset and continued class imbalance. It is not yet an empirically proven best configuration.

- Local source file: `data/financial_sentiment.csv` (2,264 rows).
- Deduplicated working DataFrame: `clean_df` (expected 2,259 rows).
- Deduplication is performed by exact sentence before any train/validation/test separation.

No model training or dataset splitting was performed on Day 1.

## 7. Limitations

- The dataset is relatively small and imbalanced, especially for the Negative class.
- High annotator agreement can favor easier examples and exclude ambiguous but realistic financial statements.
- Annotation consensus is subjective and is not proof of the actual market impact of news.
- All annotators came from the same academic institution, limiting annotation diversity.
- The corpus is based on English financial news about companies listed in Helsinki; performance on newer news, other markets, or other languages remains unverified.
- The configurations overlap and cannot be treated as independent train and test sets.
- The license restricts commercial use.

## 8. Automated Tests

Four assertion-based checks were prepared in the analysis notebook:

| Test | Evidence from analysis | Execution status |
|---|---|---|
| Label Mapping Test | Only IDs 0, 1 and 2 observed; metadata mapping confirmed | Pending final assertion output |
| Valid Text Test | No missing or empty sentences observed | Pending final assertion output |
| Duplicate Test | Five repeats identified; deduplication should yield 2,259 unique rows | Pending final assertion output |
| Class Distribution Test | All three classes present in original selection | Pending final assertion output |

Expected terminal output after running the assertions:

```text
Label Mapping Test: PASSED
Valid Text Test: PASSED
Duplicate Test: PASSED
Class Distribution Test: PASSED
```

These are expected outcomes, not a confirmed execution log. Update test status after the notebook completes all four assertions successfully.

## References

- [Financial PhraseBank dataset card](https://huggingface.co/datasets/takala/financial_phrasebank)
- [Parquet loading revision](https://huggingface.co/datasets/takala/financial_phrasebank/tree/refs%2Fpr%2F10)
- Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). [Good debt or bad debt: Detecting semantic orientations in economic texts](https://arxiv.org/abs/1307.5336). Journal of the Association for Information Science and Technology, 65.
- [CC BY-NC-SA 3.0 license](https://creativecommons.org/licenses/by-nc-sa/3.0/)
