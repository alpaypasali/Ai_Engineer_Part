# HealthBrief — Day 05: Independent Evaluation and Release Decision

**Date:** 27 November 2026  
**Project:** Week 7 / HealthBrief — V1  
**Notebook:** `notebooks/06_final_evaluation.ipynb`  
**Model:** FLAN-T5-small + LoRA (`checkpoint-96`)  
**Status:** Independent evaluation completed; research prototype documentation approved; production deployment rejected. Formal 12-test suite execution remains unverified.

> **Executive finding:** LoRA increased independent holdout ROUGE-L from 3.746519 to 14.359853. The paired mean improvement was 10.605737 points (95% bootstrap CI: 9.940739–11.230661), with higher ROUGE-L on 98 of 100 articles. However, a separate ten-article factuality audit found a supported-claim rate of only 35.42% for LoRA, 18 explicit full-article contradictions, terminology errors in all ten inspected LoRA summaries and zero fully covered key findings under a strict rubric. Better lexical overlap did not establish scientific reliability.

## 1. Objective

The final evaluation assessed whether the previously selected adapter generalizes beyond the examples used for checkpoint and decoding selection. It compared the frozen base model with the adapter-enabled model under identical inference conditions and investigated scientific accuracy, information coverage, repetition and termination. No further training or configuration optimization was performed on the final holdout.

The protocol separated three questions: whether fine-tuning improves reference overlap, whether the model faithfully communicates research findings, and whether evidence is sufficient to release the model for real users. The final holdout was reserved for the first independent estimate within this project; it is not a clinical validation cohort.

## 2. Dataset

Dataset: `BioLaySumm/BioLaySumm2025-eLife` — English biomedical research articles and plain-language reference summaries. Recorded dataset revision: `144e9c785e3a309da804eb6786feb38e6596f390`.

| Split | Original count | Project count | Purpose |
|---|---:|---:|---|
| Training | 4,346 | 500 | LoRA optimization |
| Validation | 241 | 100 | Development, loss monitoring and model selection |
| Dev-20 | Subset of validation | 20 | Generated metrics and decoding selection |
| Final holdout | Remaining original validation | 100 | Independent reference-based evaluation |
| Original official test | 142 | 100 | Sampled, not used for reference-based evaluation because references were unavailable |

The initial streaming buffer shuffle used seed 42 and buffer size 1,000. The dataset revision was recorded after initial sampling rather than explicitly pinned at that first download. The final holdout was sampled separately from the remaining validation pool. Dev-20 is a subset, not an extra split.

### Holdout integrity

| Check | Observed result |
|---|---:|
| Holdout examples | 100 |
| Unique normalized article SHA-256 IDs | 100 |
| Duplicate article IDs | 0 |
| Train–holdout intersection | 0 |
| Validation–holdout intersection | 0 |
| Non-empty reference summaries | 100 |
| Missing references | 0 |
| Manifest identity match | True |
| Integrity assertion | PASS |

The identity algorithm hashes whitespace-normalized, lowercased article text. Exact ID matching does not exclude semantic near-duplicates, related studies or exposure during base-model pretraining. The holdout was not used for project training, checkpoint selection or decoding configuration selection.

Source Selection V1 was applied without modification: recognizable Abstract and Conclusion/Discussion/Results sections are preferred, with opening text and research-result cue paragraph fallbacks. The encoder limit includes the instruction prompt and special tokens. The earlier ten-article audit retained about 3.36% of original article tokens on average and used fallback for all ten articles; this is not an estimate for all records. Final holdout preparation produced 100 valid inputs, with token lengths 511–512 (mean 511.88). References were not used in source selection.

Evidence: `dataset_split_manifest.csv`, `holdout_integrity.csv`, `source_selection_analysis.csv`.

## 3. Frozen Configuration

The final configuration was selected on development data before holdout evaluation and saved to `../configs/final_inference_config.json`. Its SHA-256 digest was recorded in `final_configuration_manifest.csv`. The full digest is available in the local manifest; the shared screenshot displayed only a prefix. A digest supports change detection, not physical immutability or verification of every weight file.

| Setting | Frozen value |
|---|---|
| Base model | `google/flan-t5-small` |
| Base revision | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| Adapter | `models/healthbrief_best_adapter/` |
| Checkpoint | `checkpoint-96` |
| Tokenizer | Saved alongside adapter |
| Source selection | V1 |
| Encoder maximum | 512 tokens |
| Training target maximum | 768 tokens |
| Final generation config | C |
| `max_length` | 256 decoder positions, including decoder start |
| `max_new_tokens` | None |
| `num_beams` | 1 |
| `no_repeat_ngram_size` | 3 |
| `repetition_penalty` | 1.0 |
| `do_sample` | False |
| `forced_eos_token_id` | None |
| `use_cache` | True |
| Evaluation batch size | 4 |
| Stored parameter dtype | FP32 |
| Final inference precision | CUDA BF16 autocast |
| Seed | 42 |
| Total adapted parameters | 77,305,216 |
| Trainable parameters during final inference | 0 |

Training provenance: 76,961,152 base parameters; 344,064 trainable LoRA parameters (0.4451% of adapted model); rank 8, alpha 16, dropout 0.05, target modules `q` and `v`, bias `none`, three epochs, 96 optimizer updates, learning rate 2e-4, micro-batch 4 and accumulation 4 (nominal effective batch 16). Training used BF16 mixed precision and no quantization. The final reload explicitly pinned the base revision; the historical training load did not explicitly pin it, so exact historical training provenance is not established.

Recorded environment: Python 3.14.4, PyTorch 2.14.1+cu130, Transformers 5.14.1, PEFT 0.22.0, Datasets 5.1.0 (training environment), CUDA 13.0, NVIDIA GeForce RTX 4060 Laptop GPU. Seed 42 improves repeatability but does not guarantee identical results across hardware or library versions.

Both models used the same base and tokenizer, with the adapter disabled for zero-shot and enabled for LoRA. The earlier tied-weight warning indicated unequal, untied `shared.weight` and `lm_head.weight` despite `tie_word_embeddings=True`; no manual retie was applied. Training precision is recorded separately from inference precision: Day 04 reproduced the Trainer ROUGE-L 13.751730 with the autocast-disabled FP32 path and manual ROUGE-L 13.274526 with BF16 autocast. Three of twenty token sequences differed between precision paths. These observations support a precision explanation for that discrepancy; no actual divergent logits were measured.

## 4. Holdout Results

Two model states generated 100 summaries each, for 200 fixed predictions. Article IDs and sample order matched for both. The generation cell took approximately 109.1 seconds in total; per-model timings were not recorded. No predictions were excluded from metric computation. Full reference summaries were used without target-length truncation.

ROUGE-1/2/L are F1 overlap scores (0–100), using stemming and the same aggregation approach as in the earlier experiments. Repeated trigram ratio is the mean of per-output ratios on a 0–1 scale after excluding special tokens; EOS and length-cap rates are percentages.

| Metric | Zero-shot C | LoRA C | LoRA minus zero-shot |
|---|---:|---:|---:|
| ROUGE-1 | 4.656008 | 26.067595 | +21.411587 |
| ROUGE-2 | 1.084190 | 5.503472 | +4.419283 |
| ROUGE-L | 3.746519 | 14.359853 | +10.613334 |
| Mean content tokens | 26.11 | 213.93 | +187.82 |
| Median content tokens | 26.50 | 233.00 | +206.50 |
| Maximum content tokens | 66 | 255 | +189 |
| Early EOS (%) | 100 | 60 | -40 pp |
| Reached maximum length (%) | 0 | 40 | +40 pp |
| Repeated trigram ratio (0–1) | 0.000400 | 0.000559 | +0.000159 |
| Empty outputs | 0 | 0 | 0 |
| Empty output rate (%) | 0 | 0 | 0 |

Differences are rounded from reported values. The 256-position decoder cap includes the start token, permitting at most 255 content tokens. Forty LoRA outputs reached that cap. Length-limit termination must not be interpreted as natural EOS termination. Diagnostic trigram rates can be nonzero despite raw-token trigram blocking because removing special tokens can create new adjacent trigrams; that explanation was not independently verified for the specific affected sequences. Low literal repetition is not evidence of semantic coherence.

Evidence: `final_holdout_metrics.csv`, `final_holdout_predictions.csv` (200 individual records; local saving was instructed, but file existence was not shown in the supplied execution outputs).

## 5. Baseline Comparison

Fine-tuning improved all three ROUGE metrics relative to the matched zero-shot baseline. It also increased mean output length and reduced the frequency of early EOS. ROUGE measures lexical overlap, not truth, causal correctness or completeness. Length differences may contribute to overlap differences; the observed effect is the end-to-end effect of enabling the trained adapter under Config C, not proof of a specific mechanism.

### Development versus holdout — LoRA Config C

| Metric | Dev-20 | Holdout-100 | Change |
|---|---:|---:|---:|
| ROUGE-1 | 26.044892 | 26.067595 | +0.022703 |
| ROUGE-2 | 5.502064 | 5.503472 | +0.001409 |
| ROUGE-L | 14.671474 | 14.359853 | -0.311621 |
| Mean generated tokens | 201.25 | 213.93 | +12.68 |
| Early EOS (%) | 75 | 60 | -15 pp |
| Length cap (%) | 25 | 40 | +15 pp |

Lexical overlap was broadly preserved on the reserved examples, whereas termination behavior deteriorated. A 0.31-point ROUGE-L decrease alone does not establish overfitting. Dev-20 was used for selection and is smaller than holdout-100; development references were tokenized/decoded while final references were complete original summaries. Differences in article topics, lengths and reference preparation can influence the comparison. No systematic distribution-shift analysis was performed on Day 05.

## 6. Statistical Analysis

Paired percentile bootstrap evaluated sampling uncertainty in the mean per-article ROUGE-L difference. Each of 1,000 iterations sampled 100 article indices with replacement using seed 42, preserving the pairing of both models' scores against the same reference. The fixed per-article F1 scores were reused; no inference or retraining occurred during bootstrap.

| Statistic | Result |
|---|---:|
| Paired articles | 100 |
| Bootstrap repetitions | 1,000 |
| Seed | 42 |
| Aggregate ROUGE-L difference | 10.613334 |
| Observed arithmetic mean difference | 10.605737 |
| Bootstrap mean difference | 10.596205 |
| Bootstrap standard error | 0.345315 |
| 95% percentile CI lower (2.5th percentile) | 9.940739 |
| 95% percentile CI upper (97.5th percentile) | 11.230661 |
| CI contains zero | False |
| LoRA wins / zero-shot wins / ties | 98 / 2 / 0 |

The interval excludes zero, supporting a positive mean ROUGE-L effect under the assumed article-sampling process. The aggregate difference uses ROUGE's internal aggregation, while the paired bootstrap targets the arithmetic mean of sample-level differences; they are deliberately reported separately. The interval does not predict each future article's score, estimate uncertainty across training runs or validate scientific accuracy. Article independence is assumed; clustered or nonrepresentative sampling could affect the interval's interpretation.

Evidence: `holdout_bootstrap.csv` (save confirmed in notebook output).

## 7. Factual Consistency

Ten articles were selected by sorting the fixed article IDs and sampling without replacement using seed 42: `18, 31, 42, 50, 55, 62, 79, 82, 93, 94`. No scores or generated texts were used as selection inputs, although aggregate results had already been seen and some predictions displayed in earlier notebook cells. This is a deterministic score-independent selection, not a fully prospectively blinded selection.

Each model's ten outputs were evaluated against the reconstructed model-visible input (maximum 512 tokens) and the full article. Re-tokenization with the same tokenizer and input settings was used to reconstruct the input; exact archived generation input IDs were not independently compared. The evaluation was AI-assisted and not independently adjudicated by a biomedical specialist. Repeated claims were not intentionally counted as new claims. There were 74 annotation records: 61 assessable claims and 13 uninterpretable records.

| Label | Definition |
|---|---|
| Supported | The evaluated source supports the claim |
| Contradicted | The evaluated source explicitly conflicts with the claim |
| Not Verifiable | The evaluated source is insufficient to verify or refute the claim |
| Uninterpretable | No coherent assessable scientific claim can be extracted |

Supported, contradicted and not-verifiable rates use assessable claims as denominator; uninterpretable records are reported separately. Rates are pooled across claims, not averaged equally over articles.

| Metric | Zero-shot input | LoRA input | Zero-shot full article | LoRA full article |
|---|---:|---:|---:|---:|
| Assessable claims | 13 | 48 | 13 | 48 |
| Supported | 8 | 17 | 8 | 17 |
| Contradicted | 5 | 17 | 5 | 18 |
| Not Verifiable | 0 | 14 | 0 | 13 |
| Uninterpretable records | 1 | 12 | 1 | 12 |
| Supported rate (%) | 61.5385 | 35.4167 | 61.5385 | 35.4167 |
| Contradicted rate (%) | 38.4615 | 35.4167 | 38.4615 | 37.5000 |
| Not Verifiable rate (%) | 0 | 29.1667 | 0 | 27.0833 |

Among the 14 LoRA input-unverifiable claims, 13 remained unverifiable and one was contradicted by the full article. None became supported. A claim absent from the input but supported by the full article would be labeled Not Verifiable / Supported; its absence from a truncated input is not evidence of falsehood. Reference summaries alone do not establish full-article factuality.

LoRA produced more supported claims in absolute terms (17 versus 8) but a lower supported proportion and more contradictions. Because it produced substantially longer outputs and more claims, counts and rates must be interpreted together. The small, AI-assisted audit is diagnostic rather than a precise estimate of population-level biomedical accuracy.

Evidence: `factuality_sample_manifest.csv`, `final_factual_consistency.csv`, `final_factuality_summary.csv`.

## 8. Key Finding Coverage

Three important findings per article were identified from the reference summaries and full articles before the qualitative output review. The same 30 findings were then applied to each model, producing 60 coverage records. Full, correct transmission of the complete predefined finding was required; partial matches were retained separately and did not contribute to strict coverage.

| Metric | Zero-shot | LoRA |
|---|---:|---:|
| Total predefined findings | 30 | 30 |
| Fully and correctly covered | 0 | 0 |
| Partially covered | 2 | 7 |
| Not covered | 28 | 23 |
| Strict coverage (%) | 0 | 0 |

Strict coverage = fully covered findings / 30 × 100. Zero strict coverage does not mean that neither model produced any correct claim; it indicates that no predefined composite finding was fully conveyed under this rubric. A short but accurate background statement can have a high supported-claim rate and still fail to summarize the actual discovery. No independent expert validation of the finding definitions or coverage labels was performed. Findings unavailable in the 512-token input can contribute to low coverage, so this measure reflects both preprocessing and generation limitations.

Evidence: `final_key_findings.csv`, `final_key_finding_coverage.csv`.

## 9. Error Analysis

Six illustrative cases were selected from the existing annotation records to represent distinct failure modes, not to estimate error frequencies. Claims and source descriptions below are paraphrases of the annotated cases. Potential causes are hypotheses, not experimentally established root causes.

| Sample / model | Error | Generated claim versus source | Potential mechanism | Potential user impact | Future mitigation |
|---|---|---|---|---|---|
| 18 / LoRA | Unsupported treatment claim | Protein structural findings presented as enabling a treatment; no treatment efficacy established | Disease–therapy association | Readers may infer a validated treatment | Source-grounded training and therapeutic claim verification |
| 42 / zero-shot | Causality reversal | Adult LepR restoration described as causing persistent obesity; early-life deficiency is the actual reported cause | Temporal or causal relation confusion | Misunderstood biological mechanism | Temporal and causal consistency checks |
| 50 / LoRA | Numerical error | RNA described as single-nucleotide rather than predominantly 26–45 nucleotides | Failure to preserve numerical attributes | Incorrect understanding of RNA precursor length | Entity-linked numerical and unit checks |
| 55 / LoRA | Terminology error | Beta-alanine transporter treated as a neurotransmitter; source identifies transporter protein | Entity-role confusion | Misinterpretation of neurotransmitter recycling | Biomedical entity-role grounding |
| 62 / LoRA | Overgeneralization | All Ctnnb1 mutants implied unable to rescue deficits; failure concerns a specific variant, while another variant rescues deficits | Omitted experimental qualifier | Invalid generalization of experimental result | Condition-aware source selection and claim checks |
| 94 / LoRA | Entity confusion | TTP assigned a transcriptional role associated with MyoD; source describes regulation of MyoD mRNA decay | Misassigned molecular role | Incorrect interpretation of muscle cell regulation | Entity–relation consistency evaluation |

### Summary-level errors (ten outputs per model)

| Category | Zero-shot | LoRA |
|---|---:|---:|
| Summaries with full-article contradictions | 3 | 8 |
| Summaries with terminology errors | 2 | 10 |
| Summaries with uninterpretable material | 1 | 10 |
| Incomplete summaries | 10 | 10 |
| Severe repetition | 0 | 0 |

A summary containing three contradicted claims counts as one affected summary but three claim-level contradictions. Scientific reversals, incorrect numerical values, entity substitutions and unwarranted treatment assertions can be consequential even if lexical overlap is high. Not-verifiable claims must not be silently counted as explicit contradictions.

Evidence: `final_error_analysis.csv`, `final_scientific_error_analysis.csv`.

## 10. Reliability Scorecard

| Category | Metric | Zero-shot C | LoRA C | Articles per model |
|---|---|---:|---:|---:|
| Summarization quality | ROUGE-L | 3.746519 | 14.359853 | 100 |
| Repetition | Repeated trigram ratio (0–1) | 0.000400 | 0.000559 | 100 |
| Termination | Early EOS (%) | 100 | 60 | 100 |
| Factuality | Input supported claim rate (%) | 61.54 | 35.42 | 10 |
| Factuality | Full-article supported claim rate (%) | 61.54 | 35.42 | 10 |
| Contradictions | Explicit full-article claims | 5 | 18 | 10 |
| Information coverage | Strict coverage (%) | 0 | 0 | 10 |
| Output quality | Empty output rate (%) | 0 | 0 | 100 |
| Output quality | Summaries with uninterpretable material | 1 | 10 | 10 |
| Output quality | Broader invalid output rate (%) | Not measured | Not measured | — |
| Efficiency | Mean per-model inference time | Not measured | Not measured | — |

A non-empty output can still be invalid or scientifically misleading. The combined generation cell duration (109.1 seconds) is not a per-model latency benchmark. Missing measurements are stored as NaN rather than fabricated values or zero. The scorecard intentionally retains different units and sample sizes and does not combine them into a single arbitrary reliability score.

Evidence: `reliability_scorecard.csv` (save confirmed).

## 11. Model Card

The research model is documented in [model_card.md](model_card.md), including the model architecture, adapter provenance, dataset and preprocessing, intended and prohibited uses, evaluation, reproduction limitations, licensing and release status.

Intended use is controlled educational and research experimentation with English biomedical summaries. Medical diagnosis, treatment advice, clinical decisions and unreviewed scientific communication are out of scope. Every generated scientific summary requires human verification. The review flag is a usage policy, not a factuality classifier.

The earlier licensing review documented Apache 2.0 for the pinned FLAN-T5-small base model. The recorded BioLaySumm dataset README did not declare an explicit dataset license; eLife articles generally have CC BY terms unless indicated otherwise, but individual rights, attribution, dataset distribution and adapter publication conditions still require verification. No separate project or adapter license has been established in the supplied evidence. Do not assume that the base model license authorizes distribution of every derivative or source-containing artifact.

## 12. Inference Package

Implementation: [`src/predict.py`](../src/predict.py), class `HealthBriefSummarizer`.

| Method | Purpose |
|---|---|
| `__init__` | Load pinned base model, saved tokenizer and frozen adapter once; validate expected configuration values |
| `validate_input` | Reject non-string or blank research text and invalid section headings |
| `prepare_input` | Apply the existing Source Selection V1 and encoder token limit |
| `generate_summary` | Run Config C under evaluation and inference mode |
| `analyze_generation` | Calculate token count, EOS, length cap, empty output and trigram repetition |
| `predict` | Return a structured summary, metadata, diagnostics and mandatory review flag |

The adapter is loaded with `PeftModel.from_pretrained(..., is_trainable=False)`. A single class instance is reused across requests, avoiding model reload for every article. CUDA uses BF16 autocast; CPU uses FP32. Unsupported CUDA BF16 fails explicitly rather than silently changing precision. The saved tokenizer and configured special token IDs are checked. Static configuration checks do not verify adapter/tokenizer file digests.

Example usage from the project root:

```python
from src.predict import HealthBriefSummarizer

summarizer = HealthBriefSummarizer()
result = summarizer.predict(article, section_headings=section_headings)
```

Here `article` is a non-empty research text and `section_headings` is its optional heading list. Results contain `summary`, `model`, `adapter`, `generation_config`, `source_token_count`, `generated_token_count`, `eos_generated`, `repeated_trigram_ratio` and additional diagnostics, including `requires_review=True` unconditionally. The flag is not proof that a particular output is incorrect, nor can a low repetition score establish safety.

The package was prepared and its syntax and selected helpers were previously reported as checked with synthetic inputs. The actual adapter and tokenizer files were not available in the report-generation environment. The user acknowledged the usage instructions, but no real inference output or formal test log was supplied. Thus real-model package execution, exact V1 source parity and CPU/CUDA end-to-end equivalence are not marked verified. Single-article inference can differ from the evaluated batch-4 CUDA path.

## 13. Automated Tests

**Formal 12-test suite status: NOT VERIFIED.** No Day 05 named test specification, test file or execution log was supplied. The following twelve evidence checks are a record of observed notebook assertions and outputs, not a replacement formal test suite. Some checks combine several assertions. No `12 passed` claim is made.

| # | Evidence check | Observed result | Status |
|---:|---|---|---|
| 1 | Frozen model parameter state | 77,305,216 total; 0 trainable | PASS observed |
| 2 | Frozen configuration record | Config C, pinned revision, manifest with SHA-256 saved | Recorded |
| 3 | Holdout count and uniqueness | 100 articles, 100 unique IDs, zero duplicates | PASS observed |
| 4 | Split separation | Zero train and validation overlap with holdout | PASS observed |
| 5 | Reference completeness and manifest match | 100 valid, 0 missing; identity match True | PASS observed |
| 6 | Source preparation | 100 inputs in correct order, 511–512 tokens | PASS observed |
| 7 | Prediction pairing | 100 per model, 200 total; article ID order matched | PASS observed |
| 8 | Generation metric record count | 200, no duplicate model/article combinations | PASS observed |
| 9 | Per-example ROUGE validity | 200 scores, 0–100 bounds; 200-row merged table | PASS observed |
| 10 | Paired bootstrap | 100 paired examples, 1,000 finite iterations, ordered CI; report saved | PASS observed |
| 11 | Factuality sample and findings integrity | 10 unique articles, 30 findings, correct manifest match | PASS observed |
| 12 | Annotation integrity | 74 claim records, 60 coverage records, 20 summary error records; allowed labels and uniqueness checks | PASS observed |

The reported package-level syntax and synthetic checks are separate from these notebook observations. Neither structural integrity checks nor passing tests can establish the correctness of scientific annotations. Obtain the Task 7 test specification and actual execution output before recording a formal pass/fail count. Do not tune the model on holdout to make a test pass.

## 14. Release Decision

| Decision | Outcome | Basis |
|---|---|---|
| Research prototype documentation | APPROVED | Controlled training, independent evaluation, statistical analysis, documented failure modes and limitations |
| Production deployment | REJECTED | Explicit scientific contradictions, terminology errors, insufficient coverage and incomplete summaries; insufficient expert validation |
| Scientifically reliable | Not established | The ten-article audit reveals material reliability problems |
| Production-ready | False | Evidence does not support autonomous scientific or clinical use |

This is a qualitative risk-based decision against the intended use, not a new numerical threshold selected after seeing the holdout. Approval of documentation does not grant permission to distribute weights or data. No new checkpoint, generation configuration or threshold was selected using final holdout results.

Future experiments are documented in [future_experiment_plan.md](future_experiment_plan.md): improved source selection, increased training examples, a larger sequence-to-sequence model and decoding optimization on new development data. Their benefits are hypotheses, not observed improvements. Any V2 informed by this evaluation must be assessed using a fresh independent holdout; the current set is no longer untouched for future design decisions.

## 15. Conclusion

HealthBrief V1 successfully completed parameter-efficient adaptation and independent evaluation. The frozen adapter improved mean sample-level ROUGE-L by approximately 10.61 points, with a positive paired bootstrap interval and improvements on 98/100 articles. Config C substantially reduced the severe repetition observed during development, yet 40% of final LoRA outputs still hit the length limit. More importantly, the small factuality audit found lower supported-claim rates for LoRA, frequent scientific contradictions, pervasive terminology problems and no fully communicated predefined key finding.

The central engineering lesson is that training success, evaluation success, scientific reliability and production readiness are distinct. The project is suitable for documenting an experimental workflow and failure analysis, not for presenting unreviewed biomedical conclusions to real users.

### Theory review — Day 05 control questions

1. **Validation versus holdout:** validation guides checkpoint and decoding selection; holdout estimates performance after those decisions are frozen. The 20 development examples were reused for selection, whereas the 100 reserved examples were not.
2. **Switching from C to B after holdout:** doing so would make the holdout influence configuration selection (evaluation leakage). A new development protocol and independent evaluation set are necessary for that revised configuration.
3. **ROUGE and scientific truth:** overlap scores cannot distinguish reduced mortality from increased mortality reliably. HealthBrief improved overlap but did not establish scientific accuracy.
4. **Not Verifiable versus Contradicted:** absent evidence is different from evidence of conflict. The former does not imply falsity.
5. **Reference-only information:** if absent from the actual input, label input-grounded factuality Not Verifiable; inspect the full article separately. Full-article support must be established rather than presumed from the reference alone.
6. **Training and inference precision:** optimization precision does not determine generation precision automatically. Day 04's three divergent token sequences show why both must be recorded.
7. **PEFT base revision:** LoRA stores low-rank updates rather than all base weights. Correct base revision, tokenizer and adapter are essential to reconstruct the model.
8. **Training success versus readiness:** completing optimization does not validate reliable use. The observed scientific errors justify the production rejection.
9. **Critical biomedical errors:** incorrect population, numeric magnitude, causal direction, molecular roles, treatment efficacy or certainty can materially distort research interpretation.
10. **Prototype versus deployment:** a research prototype may expose documented failures in controlled analysis; actual deployment requires appropriate expert validation, quality controls, monitoring, review processes and risk management.

### GitHub publication checklist

- Retain the frozen configuration, source code, safe aggregate reports and model card. Review all files before staging.
- Keep raw data, complete articles, reference summaries, source-containing review files and potentially source-reproducing predictions/evidence private until rights are verified. In particular, exclude `reports/all_factuality_articles.txt` and `reports/final_factuality_review.txt`; review notebook outputs as well as CSV contents before publication.
- Do not publish adapter weights until applicable model/data rights and attribution are confirmed. Ensure the repository does not contain secrets, tokens or private paths.
- Review the future experiment plan and record actual test execution when available. Do not claim a completed formal suite or verified live package inference without evidence.
- Stage reviewed files explicitly, check the staged diff, then commit. Repository commit/push status was not verified in this report.

### Evidence index

| Artifact | Role / availability |
|---|---|
| `day_02_peft_lora_preparation.md` | Earlier preprocessing and adapter design |
| `day_03_lora_training.md` | Training and checkpoint history |
| `day_04_peft_inference_factuality.md` | Decoding and precision experiments |
| `final_configuration_manifest.csv`, `holdout_integrity.csv` | Frozen setup and split checks; notebook save confirmed |
| `final_holdout_predictions.csv`, `final_holdout_metrics.csv` | Individual and aggregate metrics; computed tables observed, local save existence not separately confirmed |
| `holdout_bootstrap.csv` | Paired statistical evaluation; save confirmed |
| `final_factual_consistency.csv`, `final_key_findings.csv`, `final_key_finding_coverage.csv`, `final_scientific_error_analysis.csv` | Inspected annotation artifacts and notebook checks |
| `final_factuality_summary.csv`, `final_error_analysis.csv` | Notebook summary and error pattern saves confirmed |
| `reliability_scorecard.csv`, `release_decision.csv` | Notebook save confirmed |
| `model_card.md`, `future_experiment_plan.md`, `../src/predict.py` | Prepared artifacts; package real-model test not verified |

External documentation referenced in the model card: [Hugging Face Model Cards](https://huggingface.co/docs/hub/en/model-cards), [PEFT checkpoint format](https://huggingface.co/docs/peft/main/developer_guides/checkpoint), [BioLaySumm dataset](https://huggingface.co/datasets/BioLaySumm/BioLaySumm2025-eLife). This report consolidates existing project evidence; it does not claim new web license verification, model execution, or repository access.
