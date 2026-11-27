---
base_model: google/flan-t5-small
base_model_relation: adapter
library_name: peft
pipeline_tag: summarization
language: en
datasets:
  - BioLaySumm/BioLaySumm2025-eLife
tags:
  - lora
  - seq2seq
  - biomedical-summarization
  - research-prototype
---

# HealthBrief — Model Card

**Version:** V1 · **Evaluation date:** 27 November 2026  
**Status:** Research prototype — not production-ready.

HealthBrief investigates whether parameter-efficient fine-tuning can improve plain-language summaries of biomedical research. The model was successfully trained and evaluated; scientific reliability has not been established. All generated summaries require human review. This is not a clinical or scientific decision-support system.

## 1. Model Details

| Property | Value |
|---|---|
| Base model | `google/flan-t5-small` |
| Architecture | T5 encoder–decoder / sequence-to-sequence |
| Method | PEFT / LoRA, no quantization |
| Adapter | `models/healthbrief_best_adapter/` |
| Selected checkpoint | `checkpoint-96` (epoch 3) |
| Base revision used for final reload | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| Base parameters | 76,961,152 |
| LoRA parameters trained | 344,064 (0.4451% of adapted model) |
| Total parameters | 77,305,216 |
| Trainable parameters at final inference | 0 |
| Evaluated language | English |

The adapter is stored separately from the base model and needs the matching base weights and saved tokenizer. Disabling the adapter on the same base model defines the zero-shot comparison.

## 2. Intended Use

- Controlled educational and research experiments on English biomedical research summarization.
- Comparing fine-tuning, decoding strategies, evaluation metrics and model failure modes.
- Inspecting generated candidate summaries alongside original research, with human review.

## 3. Out-of-Scope Use

Medical diagnosis, treatment recommendations, clinical triage, patient-specific guidance, autonomous literature interpretation, and any unreviewed publication of generated scientific claims are out of scope. Do not substitute generated summaries for original articles, domain expertise or clinical judgment. Other languages, clinical notes and patient data have not been evaluated.

## 4. Dataset

Dataset: [BioLaySumm2025-eLife](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife). Recorded revision: `144e9c785e3a309da804eb6786feb38e6596f390`.

| Split | Original dataset | HealthBrief protocol |
|---|---:|---|
| Train | 4,346 | 500 |
| Validation | 241 | 100 for development; Dev-20 subset for generated evaluation |
| Independent final holdout | — | 100 drawn from the remaining original validation examples |
| Original test | 142 | 100 sampled; not used for reference-based final evaluation |

Initial sampling used seed 42 and streaming buffer shuffle (buffer size 1,000). The initial dataset revision was recorded after download, not pinned at the initial sampling call. The final holdout had 100 non-empty references and 100 unique normalized article SHA-256 identities, with zero exact identity overlap with train or development validation. This does not exclude near-duplicate articles or pretraining exposure. The final holdout is independent of the project's training and selection decisions, not an external clinical validation cohort.

## 5. Data Preparation

Source Selection V1 uses recognizable Abstract and Conclusion/Discussion/Results sections when available. The fallback selects opening text and a result-cue paragraph; a single-source fallback is also possible. It allocates the token budget to selected passages and accounts for prompt and special tokens. References are not used to choose input passages.

Prompt: `Summarize the following biomedical research article in plain language for a general audience:` followed by two newlines.

- Maximum encoder input: 512 tokens including prompt and special tokens.
- Training target maximum: 768 tokens; dynamic padding with label padding ignored by loss (`-100`).
- Long articles lose substantial source context. The initial 10-example audit used fallback on every example; reliable heading detection was limited.
- Final holdout inputs contained 511–512 tokens (mean 511.88).

## 6. Fine-Tuning

| LoRA setting | Value |
|---|---|
| Task | `SEQ_2_SEQ_LM` |
| Rank | 8 |
| Alpha | 16 |
| Dropout | 0.05 |
| Target modules | `q`, `v` |
| Bias | `none` |
| Adapted projections | 48 |

Only the low-rank parameters were optimized. Base parameter freezing was observed; a complete before/after tensor equality audit of all base weights was not performed.

## 7. Training Configuration

| Setting | Value |
|---|---|
| Epochs / optimizer steps | 3 / 96 |
| Learning rate | 2e-4 |
| Optimizer / scheduler | AdamW / linear |
| Train micro-batch / eval batch | 4 / 4 |
| Gradient accumulation | 4 |
| Nominal effective batch | 16 (last accumulation group smaller) |
| Weight decay / gradient clipping | 0.0 / 1.0 |
| Training precision | BF16 mixed precision |
| Seed | 42 |
| Checkpoint selection | Highest Dev-20 ROUGE-L |

Checkpoint-96 achieved Trainer Dev-20 ROUGE-L 13.751730 under the original generation settings; the full validation-100 teacher-forced loss was 2.783029. These are development results, not independent test estimates. The training-loss logging normalization discrepancy remains unresolved; raw loss values were not rescaled.

## 8. Evaluation

Both final model states used identical holdout sources, saved tokenizer, Config C, batch size 4 and CUDA BF16 autocast. All 100 examples per model were retained, including any invalid output. ROUGE-1/2/L are F1 scores on a 0–100 scale using stemming. The overall scores use the same ROUGE aggregation approach as development; the holdout uses full reference summaries, while development used tokenized/decoded references. The split comparison is therefore not perfectly matched.

| Metric | Zero-shot Dev-20 C | LoRA Dev-20 C | Zero-shot Holdout-100 C | LoRA Holdout-100 C |
|---|---:|---:|---:|---:|
| ROUGE-1 | 5.140493 | 26.044892 | 4.656008 | 26.067595 |
| ROUGE-2 | 1.484483 | 5.502064 | 1.084190 | 5.503472 |
| ROUGE-L | 4.085911 | 14.671474 | 3.746519 | 14.359853 |
| Mean generated content tokens | 27.45 | 201.25 | 26.11 | 213.93 |
| Early EOS (%) | 100 | 75 | 100 | 60 |
| Length cap reached (%) | 0 | 25 | 0 | 40 |
| Repeated trigram ratio (0–1) | 0 | 0 | 0.000400 | 0.000559 |
| Empty outputs | 0 | 0 | 0 | 0 |

LoRA improved aggregate holdout ROUGE-L by 10.613334 points. The paired sample-level arithmetic mean difference was 10.605737; 1,000 paired percentile bootstrap resamples (seed 42) produced mean 10.596205, standard error 0.345315 and 95% interval [9.940739, 11.230661]. LoRA achieved higher ROUGE-L on 98/100 articles, zero-shot on 2/100. The interval estimates article-sampling uncertainty under an independence assumption, not scientific accuracy or variability across training runs.

Mean per-model inference time and a broader invalid-output rate were not measured. The combined generation cell took approximately 109.1 seconds, which is not a model-specific latency benchmark. The zero empty-output count does not establish that outputs are valid.

## 9. Generation

Config C was selected on development results before final holdout evaluation; checkpoint and configuration were not reselected on holdout.

| Parameter | Value |
|---|---|
| Source selection | V1 |
| Max input | 512 |
| `max_length` | 256 decoder positions (including start token) |
| `max_new_tokens` | None |
| `num_beams` | 1 |
| `no_repeat_ngram_size` | 3 |
| `repetition_penalty` | 1.0 |
| `do_sample` | False |
| `forced_eos_token_id` | None |
| `use_cache` | True |
| Final evaluation precision | CUDA BF16 autocast; FP32 stored parameters |

Constrained greedy decoding reduced the development LoRA repeated trigram ratio from 0.956754 (A) to 0 (C) and increased early EOS from 10% to 75%. It does not guarantee semantic non-repetition or factual accuracy. Trigram diagnostics remove special tokens, whereas generation blocks repeats in raw token sequences, so nonzero diagnostic repetition is possible. Length-cap termination is not early natural EOS termination.

The inference package supports single-article calls and FP32 CPU inference; CPU and single-item generation were not the evaluated batch-4 CUDA setup and may differ in token outputs. CUDA BF16 is retained when running on a supported CUDA device.

## 10. Factual Consistency

Ten deterministically selected holdout articles (IDs 18, 31, 42, 50, 55, 62, 79, 82, 93, 94) were audited for both models under Config C. Selection used sorted article IDs and seed 42, without using scores, after aggregate automatic results had already been seen. Thirty key findings were defined before the qualitative review of the selected outputs.

Each claim was labeled Supported, Contradicted, Not Verifiable or Uninterpretable against both the model-visible input and the full article. Not Verifiable means the source does not establish the claim, not necessarily that the claim is false. Rates exclude uninterpretable records from the denominator while reporting them separately.

| Metric | Zero-shot C | LoRA C |
|---|---:|---:|
| Assessable claims | 13 | 48 |
| Input supported / contradicted / not verifiable | 8 / 5 / 0 | 17 / 17 / 14 |
| Full-article supported / contradicted / not verifiable | 8 / 5 / 0 | 17 / 18 / 13 |
| Input and full supported rate (%) | 61.54 | 35.42 |
| Full-article contradiction rate (%) | 38.46 | 37.50 |
| Uninterpretable records | 1 | 12 |
| Fully covered key findings / 30 | 0 | 0 |
| Partial key findings / 30 | 2 | 7 |
| Summaries with full-article contradiction / 10 | 3 | 8 |
| Summaries with terminology errors / 10 | 2 | 10 |
| Summaries with severe repetition / 10 | 0 | 0 |
| Incomplete summaries / 10 | 10 | 10 |
| Summaries with uninterpretable material / 10 | 1 | 10 |

One LoRA input-unverifiable claim became contradicted after full-article inspection; none became supported. Strict key-finding coverage excludes partial coverage. Claim count and summary count are different units. The audit used AI-assisted annotations without independent biomedical expert review or inter-annotator agreement measurement. It is exploratory and should not be generalized to a population-level factuality estimate.

Observed errors include unsupported therapeutic assertions, incorrect causal direction, numerical inconsistencies, terminology confusion, overgeneralization and entity-role errors. Greater ROUGE and longer outputs did not establish improved scientific reliability.

## 11. Known Limitations

- Limited input context, heuristic section parsing and potentially weak input–reference alignment.
- One small-model training run with 500 training articles; no seed or capacity replication.
- Severe factual/terminology problems and incomplete coverage despite lexical improvement.
- Forty percent of LoRA holdout outputs reached the generation cap; even early EOS does not establish completeness.
- No expert-reviewed release safety validation, domain-specific external test or production latency measurement.
- No evidence of adequate readability for lay audiences beyond the task setup; dedicated readability evaluation was not performed.
- Exact historical model revision pinning and full base-weight equality are not established.

## 12. Reproducibility

| Component | Recorded value |
|---|---|
| Python | 3.14.4 |
| PyTorch | 2.14.1+cu130 (recorded environment) |
| Transformers | 5.14.1 |
| PEFT | 0.22.0 |
| Datasets | 5.1.0 (recorded training environment) |
| CUDA | 13.0 |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Seed | 42 |
| Config | `configs/final_inference_config.json` |
| Config SHA-256 | Recorded in `reports/final_configuration_manifest.csv`; consult that file for the full digest |
| Tokenizer | Saved in `models/healthbrief_best_adapter/` |

The final base revision was explicitly pinned. The earlier training load did not explicitly pin the revision, so this is not proof of exact historical training provenance. Seed alone does not guarantee deterministic GPU behavior.

Reload verification reproduced the decoded summary and content-token count for a fixed validation example. The shared embedding and LM head were reported unequal and not tied despite `tie_word_embeddings=True`; no manual retie was performed. A controlled development experiment reproduced Trainer versus manual generation differences: Trainer matched the autocast-disabled FP32 path on all 20 examples, and manual inference matched BF16 autocast on all 20; 3/20 differed across these paths. Training precision and generation precision are distinct. The final evaluation retains BF16 autocast. The inference package should be verified against local notebook inputs and checkpoint files; its static configuration checks do not cryptographically establish adapter or tokenizer identity.

## 13. License

- Base model: Apache License 2.0, verified at the [pinned model revision](https://huggingface.co/google/flan-t5-small/tree/0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab). Comply with applicable license, attribution and notice obligations when redistributing.
- Dataset: no explicit dataset license field in the [recorded revision README](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife/blob/144e9c785e3a309da804eb6786feb38e6596f390/README.md). [eLife terms](https://elifesciences.org/terms) generally apply CC BY to journal articles unless indicated otherwise; individual source licenses, attribution and dataset redistribution conditions still need verification.
- HealthBrief adapter and project code: no separate project license established in the supplied evidence. Do not infer an adapter license from the base model alone. Public distribution of weights, data and source-containing reports remains pending rights review; raw articles and reference summaries should not be committed to a public repository. Generated summaries and evidence excerpts may also contain source material.

## 14. Ethical Considerations

Incorrect biomedical claims can distort research interpretation. Users must verify every summary against the original research, especially claims about treatment efficacy, population, causality, numerical results and experimental limitations. A review flag is mandatory for every prediction, regardless of EOS or repetition diagnostics. No patient-facing, clinical or autonomous scientific use is authorized by this experimental evaluation.

## 15. Release Status

| Decision | Status |
|---|---|
| Trained successfully | Yes |
| Evaluated successfully | Yes, within the documented protocol |
| Scientifically reliable | Not established |
| Research prototype documentation | APPROVED, with limitations and rights review |
| Production deployment | REJECTED |
| Production-ready | False |

No new thresholds, retraining or holdout-driven checkpoint/config changes were applied. Future source-selection, data-scaling, capacity and decoding experiments require a fresh development protocol and independent holdout. See `future_experiment_plan.md`.

### References and project evidence

- [Hugging Face model cards](https://huggingface.co/docs/hub/en/model-cards)
- [PEFT checkpoint format](https://huggingface.co/docs/peft/main/developer_guides/checkpoint)
- [BioLaySumm shared task](https://biolaysumm.org/)
- Local reports: `day_02_peft_lora_preparation.md`, `day_03_lora_training.md`, `day_04_peft_inference_factuality.md`, `final_holdout_metrics.csv`, `holdout_bootstrap.csv`, `final_factual_consistency.csv`, `final_factuality_summary.csv`, `reliability_scorecard.csv`, `release_decision.csv`.
