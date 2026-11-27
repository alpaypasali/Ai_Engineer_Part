# HealthBrief — Day 04: PEFT Inference, Decoding and Factual Consistency

**Date:** 26 November 2026  
**Project:** Week 7 / HealthBrief  
**Notebook:** `notebooks/05_peft_inference_factuality.ipynb`  
**Model:** `google/flan-t5-small` + LoRA (r=8)  
**Status:** Development experiments completed; candidate selected; formal automated test suite not verified; independent holdout not evaluated.

> **Executive finding:** Constrained greedy decoding eliminated measured trigram repetition in the 20-example development subset and improved the LoRA model's early-EOS rate from 10% to 75%, without updating model weights. However, the manual audit of five articles did not demonstrate better factual consistency or key-finding coverage. A separate controlled precision experiment reproduced the earlier Trainer/manual ROUGE-L discrepancy exactly. Config C is retained as an experimental candidate, not a scientifically reliable production model.

## 1. Objective

The objective was to reload the trained PEFT adapter in a fresh model instance, diagnose repetitive generation, assess alternative decoding configurations, and evaluate whether improvements in automatic metrics translate into better scientific summaries. The work also investigated the difference between the previous day's Trainer and manual inference ROUGE scores and selected one inference configuration before the independent final evaluation.

No additional fine-tuning, checkpoint selection, rank search, beam-search experiment, or holdout-driven tuning was performed. The scope was restricted to the existing development data.

The dataset was `BioLaySumm/BioLaySumm2025-eLife`. The previously prepared 500 training and 100 validation examples were retained. Generated metrics use the same 20 validation indices as the preceding experiments:

```text
7, 8, 12, 17, 36, 45, 47, 49, 54, 61, 63, 68, 71, 73, 75, 83, 84, 86, 89, 94
```

The manual factuality subset uses five of these indices: `7, 36, 61, 75, 94`. These are development examples, not independent tests. The reserved 100-example final holdout was not evaluated in today's workflow.

## 2. Model

The experiment uses the previously trained FLAN-T5-small encoder-decoder model and its selected LoRA adapter. The base model and the adapter are not modified during inference.

| Property | Value |
|---|---|
| Base model | `google/flan-t5-small` |
| Architecture | T5ForConditionalGeneration |
| Encoder / decoder layers | 8 / 8 |
| Model dimension | 512 |
| Vocabulary size | 32,128 |
| Base parameters | 76,961,152 |
| LoRA parameters | 344,064 |
| Total parameters with adapter | 77,305,216 |
| LoRA rank / alpha / dropout | 8 / 16 / 0.05 |
| Target modules | `q`, `v` |
| Selected checkpoint | `checkpoint-96` (epoch 3) |
| Inference trainable parameters | 0 |
| Encoder input limit | 512 tokens |
| Training label limit | 768 tokens |
| Seed | 42 |

Source Selection V1 remains unchanged. It attempts to extract informative article sections and otherwise uses an opening passage and a cue-selected paragraph. The 512-token limit includes the prompt and special tokens. Prior preparation audits showed extensive fallback use and limited source coverage; this restricts both generation quality and factuality judgments.

Zero-shot evaluation disables the adapter temporarily on the same PEFT model. LoRA evaluation enables it. This controls the base weights and tokenizer rather than comparing unrelated model architectures.

## 3. Adapter Reload

A new base model was loaded, the tokenizer was restored from `models/healthbrief_best_adapter/`, and the trained adapter was attached using `PeftModel.from_pretrained(..., is_trainable=False)`. The model was moved to CUDA and switched to evaluation mode.

### Reload verification

The previous day's saved text for validation index 7 was compared against generation from the reloaded model, preserving the original batch composition and BF16 autocast conditions.

| Check | Observed result |
|---|---|
| Validation index | 7 |
| Previous / reloaded content tokens | 767 / 767 |
| Decoded summary equality | True |
| Content-token count equality | True |
| EOS generated | False |
| Reload verification | PASS (text and token-count checks) |
| Transformers / PEFT | 5.14.1 / 0.22.0 |
| Reloaded base revision | `0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab` |
| Stored parameter dtype | `torch.float32` |
| Manual generation precision | BF16 autocast |

The reloaded output reproduced the earlier repetitive behavior. This supports successful adapter restoration for the inspected example, but historical raw output token IDs were not saved, so exact historical sequence equality cannot be claimed. The reloaded base revision matches the earlier recorded revision; the original training load did not explicitly pin it, which limits the historical provenance claim.

### Tied-weight warning

The T5 configuration indicates `tie_word_embeddings=True`, but the observed input-embedding and output-head parameters are neither tied in memory nor numerically equal (`weights_tied=False`, `weights_equal=False`). This is consistent with the earlier load warning that the checkpoint stores different values for `shared.weight` and `lm_head.weight`. No manual weight tying or checkpoint alteration was performed.

Output: `reports/adapter_reload_verification.csv`.

## 4. Generation Configuration

Three configurations were defined before evaluating their outputs. The same 20 inputs, tokenizer, adapter state, reference decoding method, batch size of four and BF16 autocast conditions were used for each applicable comparison.

| Parameter | A — Original | B — Shorter | C — Anti-Repetition |
|---|---:|---:|---:|
| `max_length` | 768 | 256 | 256 |
| `num_beams` | 1 | 1 | 1 |
| `repetition_penalty` | 1.0 | 1.0 | 1.0 |
| `no_repeat_ngram_size` | 0 | 0 | 3 |
| `do_sample` | False | False | False |
| Decoding | Greedy | Greedy | Constrained greedy |

`max_new_tokens` was explicitly cleared for the controlled decoding experiment. The length limit applies to the decoder sequence, including its start token; it is not the encoder input limit or training target limit. With no additional special tokens, the maximum content lengths are 767 and 255 respectively. The Trainer's reported `forced_eos_token_id` was `None`; a separate original-manual configuration printout is not available in the supplied evidence.

A vs B isolates the length limit. B vs C isolates trigram blocking. Beam search and repetition-penalty tuning were intentionally excluded. A repetition constraint blocks token choices rather than changing learned parameters, and it cannot guarantee scientific correctness.

## 5. EOS Analysis

Raw generated IDs were inspected rather than inferring termination from decoded text length. The first EOS position was identified, EOS-following padding was excluded, and the decoder sequence length was compared against the applicable limit. Content-token counts exclude special tokens.

### Original configuration, Dev-20

| Metric | Zero-shot | LoRA |
|---|---:|---:|
| Mean content tokens | 61.55 | 752.40 |
| Median content tokens | 25.50 | 767 |
| Maximum content tokens | 767 | 767 |
| EOS before limit | 95% (19/20) | 10% (2/20) |
| Reached maximum length | 5% (1/20) | 90% (18/20) |
| Empty decoded outputs | 0% | 0% |

The original LoRA outputs reached the limit in 18 examples, while only two produced EOS before the limit. The sample-level analysis identified those 18 as lacking EOS. A short cap does not mean the model has learned to stop naturally: Config B reached its length limit in every LoRA example. Config C increased early EOS to 15/20 for LoRA and 20/20 for zero-shot.

**Stopped by length limit is not equivalent to natural EOS termination.** An EOS at the last position must also be distinguished from a forced EOS. These checks assume the standard EOS/length stopping behavior in the present generation workflow.

## 6. Repetition Analysis

For each output, overlapping token n-grams were formed from content IDs. The metrics are computed per output and then averaged across examples (macro averaging).

- Distinct-1 = unique unigrams / total unigrams.
- Distinct-2 = unique bigrams / total bigrams.
- Repeated trigram ratio = 1 − unique trigrams / total trigrams.
- Repeated trigram count = total trigrams − unique trigrams.
- Maximum trigram frequency = highest occurrence count of any trigram in an output.

When an n-gram cannot be formed, its corresponding ratio is defined as zero, and maximum frequency is zero when no trigrams exist. This convention prevents division by zero; it does not imply good quality. These are tokenizer-level metrics, not semantic or word-level diversity measures.

| Original configuration | Zero-shot | LoRA |
|---|---:|---:|
| Mean Distinct-1 | 0.827806 | 0.032759 |
| Mean Distinct-2 | 0.931898 | 0.040897 |
| Mean repeated trigram ratio | 0.057902 | 0.956754 |
| Mean maximum trigram frequency | 3.90 | 54.85 |
| Maximum trigram frequency across outputs | 55 | 152 |

Validation index 7 provides an extreme example: LoRA produced 767 content tokens with only eight distinct token IDs; its repeated trigram ratio was 0.983007. Index 54 had only six distinct token IDs in 767 content tokens. Zero-shot was not immune: index 75 reached the cap with a repeated trigram ratio of 0.968627.

Length and repetition are related in the observed outputs, but a single long zero-shot outlier and the large number of capped LoRA outputs restrict interpretation. Distinct scores are length-sensitive. No causal conclusion or unverified numerical correlation is reported here.

Output: `reports/generation_diagnostics.csv` (40 expected per-output records).

## 7. Decoding Experiment

Two model states and three decoding configurations were evaluated on the fixed 20 inputs, producing 120 outputs. The full generation loop took approximately 51.29 seconds on the user's CUDA environment. Per-configuration runtimes were not separately measured.

| Model | Config | Avg content tokens | Distinct-2 | Repeated trigram ratio | Total repeated trigrams | Max length rate | Early EOS rate |
|---|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | A | 61.55 | 0.931898 | 0.057902 | 746 | 5% | 95% |
| Zero-shot | B | 35.95 | 0.934792 | 0.054728 | 234 | 5% | 95% |
| Zero-shot | C | 27.45 | 0.981943 | 0.000000 | 0 | 0% | 100% |
| LoRA | A | 752.40 | 0.040897 | 0.956754 | 14,366 | 90% | 10% |
| LoRA | B | 254.65 | 0.119795 | 0.872808 | 4,411 | 100% | 0% |
| LoRA | C | 201.25 | 0.760291 | 0.000000 | 0 | 25% | 75% |

### Hypothesis tests

| Hypothesis | Evidence | Outcome |
|---|---|---|
| H1: A shorter cap reduces the total number of repeated trigrams. | A→B: zero-shot 746→234 (68.63% reduction); LoRA 14,366→4,411 (69.30%). | Supported in this experiment |
| H2: Trigram blocking reduces the repetition ratio. | B→C: zero-shot 0.054728→0; LoRA 0.872808→0. | Supported |
| H3: Reduced repetition does not necessarily increase ROUGE. | B→C: ROUGE-L decreased by 0.239760 points for zero-shot and 0.145224 for LoRA despite zero measured trigram repetition. | Supported by observed scores |

All six model/hypothesis checks returned `hypothesis_supported=True`. These are descriptive checks, not statistical significance tests. Blocking trigrams can eliminate literal repetition but may allow paraphrased repetition, distorted terminology or unsupported claims. The generation paths change, not the model weights.

Output: `reports/decoding_experiment.csv` (six aggregate records; 120 generations).

## 8. ROUGE Results

ROUGE-1, ROUGE-2 and ROUGE-L were calculated using the same decoded tokenized reference summaries and stemming. Scores are on the 0–100 scale. Empty outputs were not silently excluded. ROUGE-L measures longest-common-subsequence overlap; it is not a factual entailment score.

| Model | Config | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---:|---:|---:|
| Zero-shot | A | 5.163355 | 1.379274 | 4.262540 |
| Zero-shot | B | 5.415893 | 1.401198 | 4.325671 |
| Zero-shot | C | 5.140493 | 1.484483 | 4.085911 |
| LoRA | A | 14.197240 | 1.478052 | 13.274526 |
| LoRA | B | 19.979337 | 2.353645 | 14.816698 |
| LoRA | C | 26.044892 | 5.502064 | 14.671474 |

LoRA C achieved the highest ROUGE-1 and ROUGE-2 scores; LoRA B achieved the highest ROUGE-L. C was preferred for its much better repetition behavior rather than for maximizing ROUGE-L. The gain in lexical overlap cannot establish better scientific fidelity. The references are the tokenized validation targets, not a new independent reference set.

## 9. Factual Consistency

A manual audit was performed on four outputs (zero-shot A/C and LoRA A/C) for each of the five fixed articles. The evidence source was the decoded 512-token model input, not the entire article or the reference summary. Reference summaries were used for content comparison, not as independent proof of a generated claim.

The evaluation unit was an interpretable scientific proposition. Duplicate occurrences of the same claim within a summary were counted once. Incoherent passages were recorded separately rather than silently discarded.

| Label | Definition |
|---|---|
| Supported | The selected source supports the claim. |
| Contradicted | The selected source explicitly conflicts with the claim. |
| Not Verifiable | Available input does not establish the claim. |
| Uninterpretable | No stable scientific proposition can be recovered from the passage. |

The manual annotation CSV contains 41 records across 20 summaries: 32 assessable claims and nine uninterpretable passage records. A passage record is not necessarily an entire uninterpretable summary.

| Metric | Zero-shot A | Zero-shot C | LoRA A | LoRA C |
|---|---:|---:|---:|---:|
| Supported | 4 | 4 | 1 | 6 |
| Contradicted | 1 | 1 | 0 | 1 |
| Not verifiable | 0 | 0 | 2 | 12 |
| Uninterpretable records | 1 | 1 | 2 | 5 |
| Assessable claims | 5 | 5 | 3 | 19 |
| Supported claim rate | 80.00% | 80.00% | 33.33% | 31.58% |
| Contradiction rate | 20.00% | 20.00% | 0.00% | 5.26% |
| Not-verifiable rate | 0.00% | 0.00% | 66.67% | 63.16% |

The denominator for each claim rate is Supported + Contradicted + Not Verifiable; uninterpretable records are reported separately. LoRA C produced more assessable claims and more supported claims than LoRA A, but it also produced more unsupported claims and one explicit contradiction. Its supported fraction did not improve. The small, unequal denominators prevent a strong comparative statistical conclusion.

### Representative article-level findings

| ID | Topic | Principal observation |
|---:|---|---|
| 7 | Epigenetic memory | Zero-shot reports functional FLC fusion proteins. LoRA C introduces a phosphorylated-phospholipid description conflicting with the source's fluorescent-protein fusion construction. |
| 36 | Postmortem microbial communities | LoRA C identifies the mouse model and the postmortem-interval estimation aim, but introduces unsupported brain and mollusk claims. The zero-shot human/swine description was annotated as a contradiction when interpreted as describing the current experiment; a background-context reading would make this label ambiguous. |
| 61 | Neural activity and calcium imaging | LoRA C includes supported measurement concepts but introduces an apoptosis-related claim not verified by the visible source, and misses the important quantitative detection results. |
| 75 | Genome elimination | All variants fail to convey the central findings adequately. Constrained decoding removes exact trigram repetition but introduces malformed scientific terminology and unsupported genetic claims. |
| 94 | Monoallelic gene expression | LoRA C mentions relevant active-transcription and silencing marks but introduces an unsupported chemotherapy-related statement. Zero-shot conveys the broad identification aim without sufficient coverage. |

These are source-relative annotations, not medical diagnoses or claims verified against the full literature. A Not Verifiable label must not be equated automatically with hallucination: relevant information may exist outside the selected input. Evidence excerpts and claim-level notes are retained in `reports/factual_consistency_analysis.csv` (41 rows, 30 columns).

## 10. Error Analysis

The following counts refer to unique summaries exhibiting each category, not the number of claim records. Multiple categories can apply to the same summary. Summary-level columns are aggregated after deduplicating on `(sample_id, model, config)`.

| Error category | Zero-shot A | Zero-shot C | LoRA A | LoRA C |
|---|---:|---:|---:|---:|
| Summaries evaluated | 5 | 5 | 5 | 5 |
| Unsupported scientific claim | 0 | 0 | 2 | 5 |
| Explicit contradiction | 1 | 1 | 0 | 1 |
| Missing key finding | 5 | 5 | 5 | 5 |
| Terminology error | 1 | 1 | 2 | 5 |
| Confirmed uncertainty error | 0 | 0 | 0 | 0 |
| Severe repetition | 1 | 0 | 5 | 0 |
| Incomplete summary | 5 | 5 | 5 | 5 |
| Key findings covered / defined | 1/15 | 1/15 | 1/15 | 1/15 |
| Key-finding coverage | 6.67% | 6.67% | 6.67% | 6.67% |

Three key findings per article were defined from the visible source by the manual evaluator. This rubric is not independently validated. A zero uncertainty-error count means no confirmed instance was recorded, not that uncertainty handling is reliable; some passages are uninterpretable or uncertain.

The repeated-trigram ratio fell to zero for C, and severe repetition was absent in the five audited outputs of each model under C. However, the LoRA C outputs all had at least one unsupported claim and a terminology error under the manual rubric. Neither generation constraints nor lexical overlap addressed the central coverage failure.

A plausible mechanism for repetitive text degeneration is exposure bias: the teacher-forced decoder receives reference history during training but its own generated history at inference. This can amplify an early error. The present experiment demonstrates repetition and its sensitivity to constraints; it does not establish exposure bias as the exclusive cause. Short source coverage, limited training data, model capacity and target alignment may also matter.

## 11. ROUGE Reproducibility

### Problem

Day 03 reported different ROUGE-L values for the selected checkpoint depending on the evaluation path: Trainer = 13.751729578646366; manual inference = 13.274526093657549. The difference was 0.477203484988817 points. The purpose of the investigation was to reproduce the discrepancy rather than manually change reported scores.

A new `Seq2SeqTrainer` was created on the reloaded checkpoint and `predict()` was used on the fixed development subset. The historical Trainer metrics were reproduced, including loss 2.763699, ROUGE-1 14.778891, ROUGE-2 1.685329 and ROUGE-L 13.751730. No optimizer updates were performed.

| Control | Observed result |
|---|---|
| Input IDs, sample order and attention masks | Match across 20 examples |
| Decoded references | Match across 20 examples |
| Trainer evaluation mode | True |
| Trainer BF16 argument | True (configuration only) |
| Five explicit generation parameters | Match |
| Trainer forced EOS / max new tokens | None / None |
| Trainer use cache | True |
| Generated sequences after padding normalization | 17/20 match |

The first EOS and post-EOS padding were used to normalize generation sequences; no EOS-free content tokens were removed. The first divergences were:

| Validation ID | First differing position (zero-based, including decoder start) | Trainer token ID | Manual token ID |
|---:|---:|---:|---:|
| 45 | 2 | 3 | 454 |
| 49 | 5 | 8 | 3 |
| 75 | 13 | 10363 | 9516 |

### Controlled precision experiment

A fresh model instance with the same recorded base revision and adapter was evaluated twice using identical batches and Config A. The experimental variable was CUDA BF16 autocast (disabled vs enabled). The FP32 label here means float32 stored parameters with BF16 autocast disabled; other GPU math settings such as TF32 were not altered.

| Comparison | Matching sequences | Match rate |
|---|---:|---:|
| Trainer vs manual | 17/20 | 85% |
| Trainer vs FP32 path | 20/20 | 100% |
| Trainer vs BF16 path | 17/20 | 85% |
| Manual vs FP32 path | 17/20 | 85% |
| Manual vs BF16 path | 20/20 | 100% |
| FP32 vs BF16 | 17/20 | 85% |

All four output sets were scored using a shared ROUGE function, the same decoded references and a controlled NumPy seed for aggregation.

| Method | Recomputed ROUGE-L |
|---|---:|
| Trainer | 13.751730 |
| Manual | 13.274526 |
| FP32 path | 13.751730 |
| BF16 path | 13.274526 |

The reported Trainer/FP32 and manual/BF16 ROUGE-match checks both returned True. The exact score and token-sequence reproduction strongly supports **generation precision mismatch** as the cause of the discrepancy. The directly instrumented autocast-state column was not visible in the shared truncated table, so its actual value is not independently verified here. Historical Day 03 raw Trainer token IDs were also unavailable; the token-level comparison uses the reconstructed Trainer run. Actual divergent-token logits were not measured.

Numerical precision can change the ranking of close token scores. Greedy decoding makes a discrete selection at each step; a single changed token can alter all subsequent conditioning. The observation does not imply that FP32 is universally better than BF16.

### Resolution

Use consistent inference precision within each comparison. To reproduce the Trainer generation path in this environment, use the controlled autocast-disabled path. The existing A/B/C experiments remain valid as matched BF16-autocast comparisons; their scores are retained without modification. For the final candidate evaluation, retain the BF16-autocast path on which the selection was made, and disclose that precision in the report.

Output: `reports/rouge_reproducibility_analysis.csv` (20 per-example rows, with repeated aggregate ROUGE fields).

## 12. Candidate Selection

The model checkpoint remains `checkpoint-96`. The selection concerns only the inference configuration. The shared automatic/manual comparison contains four rows (A/C for the two model states) and 31 columns, with exactly one candidate marked for final evaluation. Config B is retained in the decoding experiment but excluded from the joint manual comparison because no factuality audit was performed for it.

The decision follows the stated priorities, with an explicit trade-off: Config C does not win on the first priority (explicit scientific contradictions), so it cannot be represented as a strict lexicographic winner. Config A has zero observed contradictions but only three assessable claims and severe repetition in all five manually inspected summaries. Config B has no manual factuality evidence and remains highly repetitive. The experimental candidate is C because A is largely unusable as a summary, while C retains observed key-finding coverage and greatly improves repetition and EOS behavior. The contradiction introduced by C remains a blocking concern for deployment.

| Criterion | Assessment |
|---|---|
| Scientific contradictions | LoRA A: 0; LoRA C: 1. C has not improved this criterion. |
| Severe repetition | LoRA A: 5/5; LoRA C: 0/5. |
| Key findings | Both cover 1/15 under the manual rubric. |
| ROUGE | C leads on ROUGE-1/2; B has slightly higher ROUGE-L than C. |
| Inference cost | C produces 201.25 content tokens on average, compared with 752.40 for A and 254.65 for B. Per-config latency was not measured. |

### Frozen candidate for Friday, 27 November 2026

| Setting | Selected value |
|---|---|
| Model state | LoRA enabled |
| Adapter location | `models/healthbrief_best_adapter/` |
| Training checkpoint | `checkpoint-96` |
| Source selection | V1, 512-token encoder input |
| `max_length` | 256 |
| `max_new_tokens` | None |
| `num_beams` | 1 |
| `repetition_penalty` | 1.0 |
| `no_repeat_ngram_size` | 3 |
| `do_sample` | False |
| Precision | BF16 autocast on CUDA; FP32 stored weights |
| Seed | 42 |
| Development batch size | 4 |
| Selection scope | Development validation only |

The candidate must not be retuned after seeing the reserved holdout. A future change requires a separately documented experiment and an independent evaluation set. This decision is not a production readiness approval and does not imply clinically safe or scientifically faithful output.

Output: `reports/generation_quality_comparison.csv` (4 rows × 31 columns; one selected candidate).

## 13. Limitations

- The 20 development examples were reused for checkpoint and decoding selection. They are not independent test observations; the five-article factuality subset is smaller still.
- Claim annotation is manual and single-evaluator. Claim segmentation, ambiguous context and key-finding selection may affect results. No inter-rater agreement was measured.
- The model receives only selected passages within 512 encoder tokens; evidence outside that input cannot establish source-relative support. Source selection V1 frequently fell back in earlier audits.
- FLAN-T5-small was adapted with just 500 examples and a single LoRA configuration. The experiment does not identify the best model architecture, rank or training setup.
- ROUGE measures lexical overlap rather than scientific validity; Distinct scores are length-sensitive and trigram blocking does not prevent semantic redundancy.
- C's contradiction and terminology issues, low supported-claim rate and 6.67% key-finding coverage remain serious quality limitations.
- Precision effects were reproduced for one environment and one fixed development subset, not across arbitrary GPUs or software versions. The historical raw token sequences and visible direct autocast instrumentation are incomplete.
- The original training revision was not explicitly pinned at the final model-loading call. The reloaded revision matches the recorded revision but does not prove exact historical provenance.
- The total generation runtime is known, but per-configuration latency and peak memory for this decoding comparison were not measured. Output length is a cost proxy only.
- The original official test sample lacks accessible reference summaries, and dataset redistribution permissions were not confirmed in prior work. The reserved reference-bearing holdout remains separate.

## 14. Automated Tests

**Formal suite status: NOT VERIFIED.** The requested ten automated tests were not supplied as a named test specification or an execution log. Therefore, the following ten documented notebook checks are an evidence inventory, not a claimed ten-test automated-suite result. Some are explicit assertions and others are observed comparisons. No pytest execution or pass count for a formal suite is claimed.

| # | Documented notebook check | Observed result | Status |
|---:|---|---|---|
| 1 | Reloaded adapter text and content-token count match previous example | True / True (767 tokens each) | Observed PASS |
| 2 | Inference trainable parameter count | 0 | Observed PASS |
| 3 | Generation output count per model, original diagnostics | 20 zero-shot and 20 LoRA | Observed PASS |
| 4 | Decoding experiment total output count | 120 (assertion completed) | Observed PASS |
| 5 | Trainer/manual input IDs and attention masks | True / True | Observed PASS |
| 6 | Trainer evaluation mode | True | Observed PASS |
| 7 | Decoded reference equality | True for all 20 | Observed PASS |
| 8 | Generated sequence comparison | 17/20 equal; three differences located | Diagnostic result; equality not satisfied |
| 9 | Precision-matched sequence and ROUGE comparisons | Trainer/FP32 and manual/BF16: 20/20 and ROUGE match=True | Observed PASS |
| 10 | Candidate selection uniqueness | One selected row out of four (assertion completed) | Observed PASS |

The unequal sequences in check 8 are the investigated behavior, not an unexplained failure: the controlled precision experiment reproduces the difference. Exact specifications and actual execution output of the requested ten formal tests must be obtained before the formal suite can be marked complete.

## 15. Conclusion

The PEFT reload preserved the previous model's observed generation behavior. The original LoRA checkpoint achieved better lexical-overlap scores than the zero-shot model, but was prone to severe text degeneration and failure to terminate before the length limit. Shortening the cap reduced total repetition without fixing EOS behavior. Trigram blocking removed measured trigram repetition and substantially improved early termination without retraining.

These improvements did not establish scientific reliability. The five-article audit revealed unsupported claims, terminology errors, an explicit contradiction, and poor coverage of core findings. The project illustrates why a biomedical summarization system must be evaluated on factual support, contradictions, information coverage and generation behavior in addition to ROUGE.

The controlled precision experiment also demonstrated that nominal training precision configuration alone does not establish generation precision equivalence. Reproducing token sequences under matching precision was essential for explaining the Day 03 score discrepancy.

Config C with the fixed LoRA checkpoint and BF16 autocast is the experimental candidate for Friday's reserved holdout. No holdout-driven tuning is permitted. The current system is not suitable for unsupervised scientific or clinical use.

### Repository deliverables

```text
HealthBrief/
├── notebooks/
│   └── 05_peft_inference_factuality.ipynb
└── reports/
    ├── adapter_reload_verification.csv
    ├── generation_diagnostics.csv
    ├── decoding_experiment.csv
    ├── factual_consistency_analysis.csv
    ├── rouge_reproducibility_analysis.csv
    ├── generation_quality_comparison.csv
    └── day_04_peft_inference_factuality.md
```

The notebook and reports above are the intended local repository outputs; the current review used shared notebook results and the supplied annotation/reproducibility data rather than a full repository checkout. The generation diagnostics CSV save was instructed but not independently confirmed from a file listing. The formal test suite and GitHub commit/push are not claimed as completed. Before publishing, review the dataset and model license terms and whether CSVs containing full source excerpts or reference summaries may be redistributed.
