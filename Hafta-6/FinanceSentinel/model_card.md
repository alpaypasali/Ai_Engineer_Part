# FinanceSentinel — Model Card

**Author:** Alpay Paşalı  
**Date:** 2026-11-20  
**Version:** 1.0 (experimental)

FinanceSentinel is a fine-tuned DistilBERT model for classifying English financial news sentences as Negative, Neutral, or Positive. I built this project to explore not only classification performance, but also model confidence, calibration, inference cost, and the limitations of a small financial sentiment dataset.

The final model achieved a test Macro-F1 of 0.938100. However, targeted robustness tests revealed consistent semantic errors, so I consider this model a research prototype rather than a production-ready financial decision system.

## 1. Model Details

| Property | Value |
|---|---|
| Base model | [`distilbert-base-uncased`](https://huggingface.co/distilbert/distilbert-base-uncased) |
| Architecture | DistilBERT for sequence classification |
| Language | English |
| Transformer layers | 6 |
| Hidden size | 768 |
| Attention heads | 12 |
| Total parameters | 66,955,779 |
| Fine-tuning method | Full fine-tuning |
| Weight format | Safetensors |
| Maximum input length | 256 tokens, including special tokens |
| Selected checkpoint | `models/length_resource_experiment_256/checkpoint-339` |
| Final model directory | `models/financesentinel_final/` |

The classifier produces three logits for each sentence. Softmax converts these logits into normalized scores, and the highest-scoring class becomes the predicted label.

| Label ID | Class |
|---:|---|
| 0 | Negative |
| 1 | Neutral |
| 2 | Positive |

The original checkpoint and final model export were compared on 20 fixed validation examples. Logits and probabilities matched within numerical tolerance (`atol=1e-6`, `rtol=1e-5`), and predicted labels matched exactly. The recorded save/reload test passed.

| Exported file | Size (MB) |
|---|---:|
| `config.json` | 0.0008 |
| `tokenizer.json` | 0.7115 |
| `tokenizer_config.json` | 0.0005 |
| `model.safetensors` | 267.835644 |
| Total folder | 268.548396 |

The final checkpoint uses FP32 weights. These sizes were recorded in the final evaluation notebook; they are not estimates.

## 2. Intended Use

The model is intended for non-commercial research, education, and experimental analysis of short English financial news sentences. It can help organize documents by sentiment, explore historical financial language, and support human review of large collections of financial statements.

It is most appropriate for sentences similar to the training corpus and where incorrect predictions can be reviewed by a person. Its output should be interpreted as a text classification result, not a forecast of financial performance or market movements.

## 3. Out-of-Scope Use

The model should not be used for:

- Investment recommendations, automated trading, credit decisions, or other high-stakes financial decisions.
- Stock-price prediction or assessment of a company's future financial performance.
- Unsupervised classification in high-risk workflows where mistakes can cause financial harm.
- Reliable interpretation of full financial reports, long documents, non-English text, or unfamiliar financial domains without separate evaluation.
- Commercial deployment without verifying the applicable dataset and model rights.

## 4. Training Data

The model was trained on the `sentences_allagree` configuration of [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank). I selected this configuration because of its high annotator agreement. Five duplicate sentences were removed from the original 2,264 examples, resulting in 2,259 unique sentences.

Two stratified splits with seed 42 were used. The same splits were maintained throughout model selection and final evaluation.

| Split | Samples | Negative | Neutral | Positive |
|---|---:|---:|---:|---:|
| Train | 1,807 | 242 | 1,109 | 456 |
| Validation | 226 | 30 | 139 | 57 |
| Test | 226 | 31 | 138 | 57 |
| Total | 2,259 | 303 | 1,386 | 570 |

Neutral is the majority class. Macro-F1 was therefore selected as the primary evaluation metric, supported by class-specific recall and precision. Exact duplicates were checked before splitting; near-duplicate and source-level overlap were not fully investigated.

More information is available in the [Data Card](data_card.md).

## 5. License

The pretrained DistilBERT model is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Financial PhraseBank is distributed under [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/), which imposes attribution, non-commercial and share-alike conditions on the licensed material and applicable adaptations.

These are separate licenses. The base model's Apache license does not automatically resolve the rights associated with the training data or the distribution of fine-tuned weights. FinanceSentinel is currently an educational and non-commercial research project. Before redistributing the trained checkpoint, raw financial sentences or derived data, or deploying the model commercially, the applicable rights and permissions must be verified. I keep raw data and model checkpoints local by default.

## 6. Training Configuration

| Hyperparameter | Value |
|---|---|
| Epochs | 3 |
| Learning rate | 2e-5 |
| Training batch size | 16 |
| Evaluation batch size | 16 |
| Weight decay | 0.01 |
| Warmup ratio | 0.1 |
| Learning rate scheduler | Linear |
| Gradient accumulation steps | 1 |
| Maximum sequence length | 256 |
| Padding | Dynamic |
| Truncation | Enabled |
| Best model metric | Macro-F1 |
| Seed | 42 |
| Global training steps | 339 |

The model was fine-tuned using Hugging Face Transformers and PyTorch. The selected checkpoint was obtained at epoch 3 and exported together with its tokenizer for reuse.

## 7. Model Selection

I compared two sequence-length configurations while keeping the validation split fixed.

| Max Length | Validation Accuracy | Validation Macro-F1 | Training Time (s) | Peak GPU Memory (MB) |
|---:|---:|---:|---:|---:|
| 128 | 0.964602 | 0.949958 | 18.043863 | 1514.97 |
| 256 | 0.969027 | 0.958797 | 18.087495 | 1633.62 |

The 256-token model was selected because it achieved the higher validation Macro-F1, with a small increase in training time and a moderate increase in GPU memory. All validation sentences were shorter than 128 tokens, so the observed improvement cannot be attributed directly to retaining more validation text. The comparison was based on individual training runs, not a multi-seed experiment.

The independent test set was reserved until after model selection; its results were not used to choose the checkpoint.

## 8. Evaluation

| Metric | Validation | Independent Test |
|---|---:|---:|
| Samples | 226 | 226 |
| Accuracy | 0.969027 | 0.951327 |
| Macro-F1 | 0.958797 | 0.938100 |
| Macro Precision | 0.950717 | 0.930442 |
| Macro Recall | 0.967601 | 0.946458 |
| Negative Recall | 0.966667 | 0.935484 |
| Neutral Recall | 0.971223 | 0.956522 |
| Positive Recall | 0.964912 | 0.947368 |
| Incorrect predictions | 7 | 11 |

The test Macro-F1 is approximately 2.07 percentage points below the validation result. This is encouraging for generalization to the held-out portion of the dataset, but it does not establish performance on new financial domains or current news.

### Statistical uncertainty

The test Macro-F1 was evaluated using 1,000 paired bootstrap resamples with replacement and seed 42.

| Bootstrap measure | Result |
|---|---:|
| Test Macro-F1 | 0.938100 |
| Bootstrap mean | 0.937704 |
| 95% percentile confidence interval | [0.897937, 0.973983] |

The interval reflects sampling uncertainty within this test set. It does not measure uncertainty from retraining the model or changes in the data distribution.

![Validation and test comparison](../figures/validation_test_comparison.png)

![Test confusion matrix](../figures/test_confusion_matrix.png)

## 9. Calibration

I measured calibration using five equal-width probability bins. The overall ECE uses the maximum predicted probability and prediction correctness. Class-wise ECE is computed using all validation examples in a one-vs-rest setup.

| Validation calibration metric | ECE |
|---|---:|
| Overall ECE | 0.005165 |
| Negative | 0.010792 |
| Neutral | 0.024971 |
| Positive | 0.017487 |

The independent test overall ECE was **0.020372** (five bins), compared with 0.005165 on validation, an increase of 0.015207. Recalculation from the 226 saved test predictions agrees to six decimal places. The class-wise values above refer to validation only.

The average confidence of correct validation predictions was 0.968019, compared with 0.843052 for incorrect predictions. Four validation mistakes and six test mistakes were made with confidence of at least 90%.

The low validation ECE should not be mistaken for a guarantee that the model is reliable on individual predictions. It depends on the sample size and binning approach. No temperature scaling was fitted. The higher test ECE suggests a larger calibration gap on the held-out split, though the sample is too small to establish a general calibration difference.

![Overall reliability diagram](../figures/reliability_diagram.png)

![Class-wise calibration](../figures/classwise_calibration.png)

## 10. Inference Cost

I benchmarked the final model on the same 226 validation sentences, using one device, five warm-up calls per repeat, three repeats per configuration, dynamic padding, and CUDA synchronization. Timing includes tokenization, transfer to the device, model inference, softmax and transfer of predictions and confidence scores back to the CPU. Model loading is excluded.

| Batch Size | Mean Batch Latency (ms) | P95 Batch Latency (ms) | Throughput (samples/s) | Peak Allocated GPU Memory (MB) |
|---:|---:|---:|---:|---:|
| 1 | 1.6581 | 1.9150 | 600.9045 | 279.7471 |
| 8 | 6.5685 | 9.1267 | 1184.8242 | 297.2237 |
| 16 | 11.5064 | 14.3045 | 1308.6036 | 314.0326 |
| 32 | 21.2115 | 26.7227 | 1331.3003 | 352.1905 |

Batch 1 has the lowest measured latency. Batch 32 has the highest measured throughput, but its improvement over batch 16 is only about 1.73%, while mean batch latency increases substantially. Batch 16 is a reasonable balance for offline processing; batch 1 is a reasonable starting point for individual requests. These are local benchmark results, not production load-test guarantees. They exclude network and request-queue delays.

Under comparable conditions, processing one million sentences at batch 32 is estimated to take around 12.52 minutes. The selected model took 18.09 seconds to train and used 1633.62 MB of peak allocated GPU memory. The training and inference costs are tied to the local hardware and workload.

![Inference latency and throughput](../figures/latency_throughput_comparison.png)

## 11. Error Analysis and Robustness

| Error category | Validation | Test |
|---|---:|---:|
| Cost reduction misinterpretation | 2 | 0 |
| Neutral classified as Positive | 4 | 4 |
| Negative classified as Positive | 1 | 2 |
| High-confidence errors (≥ 0.90) | 4 | 6 |
| Possible truncation-related errors | 0 | 0 |
| Total errors | 7 | 11 |

Categories can overlap and should not be summed. The cost-reduction count is zero in the test set after manual review of the 11 errors; the notebook's earlier `NaN` was a placeholder, not an observed error count. The test set does include a distinct case where risk reduction was incorrectly interpreted as Negative. Loss increases were sometimes classified as Positive. These errors suggest that the model may associate directional terms with sentiment without consistently accounting for what is increasing or decreasing.

I tested this hypothesis using ten manually labeled synthetic financial sentences. The expected labels were defined before predictions were generated.

| Scenario | Expected | Predicted | Confidence |
|---|---|---|---:|
| Operating costs decreased | Positive | Negative | 0.969307 |
| Operating costs increased | Negative | Positive | 0.974218 |
| Operating profit decreased | Negative | Negative | 0.970819 |
| Operating profit increased | Positive | Positive | 0.976044 |
| Negation / unchanged profit | Neutral | Positive | 0.807488 |
| Loss reduction | Positive | Negative | 0.970356 |
| Debt increase | Negative | Positive | 0.973829 |
| Future growth target | Neutral | Positive | 0.977778 |
| Actual growth | Positive | Positive | 0.978089 |
| Mixed sentiment | Neutral | Negative | 0.951991 |

Only three of ten examples matched their expected labels (30%); six of seven incorrect predictions exceeded 90% confidence. This targeted test is too small and selective to estimate overall accuracy, and mixed sentiment is inherently debatable. Nevertheless, the controlled cost/profit examples provide evidence of a potential lexical shortcut and expose important weaknesses in negation, financial direction and the distinction between planned and realized outcomes.

## 12. Limitations

- The training corpus is small, English-only, imbalanced and restricted to sentences with high annotator agreement.
- The test split comes from the same source as the training data. External, temporal and cross-domain generalization remain unverified.
- The model may confuse cost, debt, loss and profit changes and may overlook negation or mixed sentiment.
- High confidence is not a guarantee of correctness; a simple confidence threshold cannot eliminate all serious errors.
- Inputs longer than 256 tokens may lose important context due to truncation.
- Only individual training runs were compared; robustness across random seeds and hardware has not been established.
- The ten synthetic challenges are diagnostic examples rather than a representative benchmark.

## 13. Ethical Considerations

Financial sentiment describes the language of a statement, not whether an investment is suitable or likely to succeed. A confidently wrong Positive prediction about rising losses or debt could mislead someone reading a financial news summary. Incorrect Negative predictions could also create unnecessary alerts or unfairly distort the description of a company's performance.

Any experimental use should include transparent uncertainty reporting, access to the original context, and human review for consequential decisions. Dataset licensing, attribution and the rights associated with underlying news texts must be respected. The model is not intended to replace qualified financial analysis.

## 14. Monitoring

If the model is evaluated in a controlled deployment, I would monitor:

| Monitoring area | Signals |
|---|---|
| Classification quality | Macro-F1, per-class precision and recall using independently reviewed labels |
| Calibration | ECE, confidence distribution and confidently wrong predictions |
| Data drift | Changes in token lengths, vocabulary, document sources and input topics |
| Label and concept drift | Class prevalence and changes in the input-to-sentiment relationship using fresh labels |
| Robustness | Recurring failures on negation, cost/profit direction and mixed sentiment |
| Operational performance | P50/P95 latency, throughput, error rate and GPU memory |

I would keep versioned checkpoints, maintain a human-review workflow, repeat the robustness suite on new versions and evaluate any retrained model on fresh holdout data before release. Monitoring input drift alone is not sufficient to establish performance degradation; recent labeled examples are needed.

## 15. Final Assessment

FinanceSentinel is a useful research prototype for short English financial sentiment classification. The independent test results are strong relative to the small dataset, and inference is efficient on the measured hardware. The robustness results, however, are a direct reminder that a high Macro-F1 does not imply reliable financial reasoning.

I would use the model for educational experiments and human-reviewed exploratory analysis. I would not deploy it as an autonomous system for high-stakes financial decisions. Broader labeled data, independent domain evaluation, systematic counterfactual testing, and appropriate rights clearance are required before considering higher-risk use.

## 16. Verification Status

All 12 automated tests passed in the final evaluation notebook on 2026-11-20, as confirmed by `final_test_results.csv`. These include model reload, tokenizer and prediction consistency, logits, split integrity, exact data leakage, probability and metric validity, latency validity, batch consistency, error consistency and ECE validity. On 2026-11-22, standalone fresh-process model and benchmark checks passed, with peak throughput of 868.91 samples/s (batch 32). The separate application pytest suite also passed all 12 tests in 2.89 seconds.

The application flags predictions below 0.80 confidence and all truncated inputs for manual review. In the validation threshold analysis, 0.80 achieved 94.25% coverage and 98.12% accepted accuracy (13 rejections out of 226). This policy is not a safety guarantee: all seven counterfactual errors exceeded the selected threshold.

![Fresh-process verification](../figures/final_model_verification.png)

## References and Evaluation Artifacts

- [DistilBERT base model](https://huggingface.co/distilbert/distilbert-base-uncased)
- [Financial PhraseBank dataset](https://huggingface.co/datasets/takala/financial_phrasebank) — Malo et al. (2014), *Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts*.
- [Day 04 calibration and error analysis](day_04_calibration_error_analysis.md)
- [Final test metrics](final_test_metrics.csv)
- [Bootstrap results](bootstrap_results.csv)
- [Inference benchmark](inference_benchmark.csv)
- [Robustness analysis](robustness_analysis.csv)
- [Automated test results](final_test_results.csv)
- [Final error comparison](final_error_comparison.csv)
- [Threshold analysis](threshold_analysis.csv)
