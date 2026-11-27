# HealthBrief — Future Experiment Plan (V2)

**Status:** Proposed experiments only; not executed.  
**Current version:** FLAN-T5-small + LoRA, checkpoint-96, Source Selection V1, Config C, BF16 autocast, seed 42.  
**Release decision:** Research/educational documentation approved; production deployment rejected. Public distribution remains subject to license verification.

## 1. Motivation and baseline

The current model improves lexical overlap but remains unsuitable for reliable automated biomedical summarization. The independent holdout contained 100 articles; the AI-assisted qualitative review covered 10 of those articles (10 outputs per model).

| Metric | Zero-shot C | LoRA C | Scope |
|---|---:|---:|---|
| ROUGE-L (0–100) | 3.746519 | 14.359853 | 100 articles |
| Early EOS rate | 100% | 60% | 100 articles |
| Max length rate | 0% | 40% | 100 articles |
| Repeated trigram ratio (0–1) | 0.000400 | 0.000559 | 100 articles |
| Input-supported claim rate | 61.54% | 35.42% | 10 articles |
| Full-article supported claim rate | 61.54% | 35.42% | 10 articles |
| Full-article contradicted claims | 5 | 18 | 10 articles |
| Strict key finding coverage | 0/30 | 0/30 | 10 articles |
| Summaries with terminology errors | 2/10 | 10/10 | 10 articles |

The paired mean ROUGE-L difference was 10.605737 points (1,000 percentile bootstrap resamples, seed 42; 95% CI [9.940739, 11.230661]). This interval quantifies uncertainty in mean lexical-overlap improvement under the article-sampling assumptions, not factual accuracy or training variability. Per-model inference latency and comprehensive invalid-output rate were not measured. The qualitative labels were not independently validated by biomedical experts; strict coverage excludes partial matches.

## 2. H1 — Source selection is insufficient

**Hypothesis:** V1 may omit relevant experimental results and qualifiers from the model's 512-token input. Missing information may limit coverage and contribute to ungrounded generation.

**Experiment:** Compare V1 against a section-aware V2 that prioritizes abstract, central results and conclusions, with an audited fallback for missing headings. Hold the 512-token encoder budget constant initially. Record selected sections, fallback frequency and whether each predefined key finding is present in the input. Do not use reference summaries to select model inputs.

**Controls:** Same frozen adapter, base revision, tokenizer, Config C, precision and evaluation articles for the initial inference-only ablation. If retraining with V2 is needed because of input-distribution shift, run it as a separate matched experiment. A larger input budget is also a separate experiment.

**Expected benefit:** Better access to key findings and fewer omitted conditions; factuality improvement is a hypothesis, not a guaranteed consequence.

**Cost:** Medium; preprocessing implementation, source audits and repeated inference. Retraining or longer inputs increase cost.

**Validation:** Input key finding availability, input-grounded and full-article supported/contradicted rates, strict key finding coverage, ROUGE and input retention. Compare identical articles and preserve source evidence.

## 3. H2 — Training data is insufficient

**Hypothesis:** Training on 500 examples may not provide enough high-quality supervision for diverse biomedical terminology and relations. Increasing sample count alone may not help if source-target alignment is poor.

**Experiment:** Compare nested training subsets of 500, 1,000, 2,000 and, if eligible, 4,000 articles. Use only authorized training records and deduplicate against all evaluation sets. Audit source-target alignment and record dataset revisions and subset IDs.

**Controls:** Same base model, tokenizer, preprocessing, LoRA settings, generation settings and evaluation protocol. Hold learning rate, effective batch size and epoch policy constant for the primary data-scale experiment. Record optimizer updates, tokens and compute; fixed epochs imply more updates for larger datasets. If feasible, add a matched-update control to separate data-volume effects from training-budget effects. Repeat with multiple seeds when resources permit.

**Expected benefit:** Better scientific entity/relation preservation, higher coverage and fewer terminology errors.

**Cost:** High; training time and compute increase with the amount of data and training steps; data audits require additional effort.

**Validation:** Claim-level factuality, terminology error count/rate, coverage, ROUGE, EOS, training curves, runtime and peak GPU memory. Compare uncertainty across training seeds where feasible.

## 4. H3 — Model capacity is limited

**Hypothesis:** FLAN-T5-small may lack sufficient capacity to model complex biomedical entities and relationships under the present setup.

**Experiment:** Compare FLAN-T5-small with FLAN-T5-base; consider FLAN-T5-large only if a hardware feasibility test succeeds. Train a separate compatible LoRA adapter for each architecture. Do not transfer the existing small-model adapter to a differently shaped base model.

**Controls:** Same eligible training and development articles, source-selection policy, input/target limits, decoding settings and factuality rubric. Record each model's revision and tokenizer, actual LoRA parameter count, training budget and precision. Where hardware forces different settings, disclose them and avoid attributing all differences solely to capacity.

**Expected benefit:** Better contextual representation and preservation of scientific claims, terminology and qualifiers.

**Cost:** High; increased memory, training and inference requirements. The existing RTX 4060 Laptop GPU must pass a memory feasibility check before larger-model training.

**Validation:** Supported and contradicted claim rates, summary-level contradiction incidence, coverage, ROUGE, EOS, trainable/total parameters, peak VRAM and latency.

## 5. H4 — Decoding remains problematic

**Hypothesis:** Length limits and token-level repetition constraints reduce loops but do not guarantee natural EOS, complete summaries or accurate claims.

**Experiment:** Use Config C as the baseline on new development data. Test one factor at a time initially: max_length (256/384/512), num_beams (1/2/4), repetition_penalty (1.0/1.1/1.2), no_repeat_ngram_size (0/3/4). Test length_penalty (0.8/1.0/1.2) in a separate beam-search comparison; it is not a meaningful greedy-decoding comparison. Keep do_sample=False and record effective generation settings, including EOS behavior. Treat these values as proposed candidates, not selected settings.

**Controls:** Same frozen model, tokenizer, source selection, inputs, BF16 autocast and batch size. Keep other generation parameters fixed in each ablation. Use synchronized timing with warm-up and record model-specific batch throughput and latency separately.

**Expected benefit:** Fewer truncated summaries, better natural termination, and less repetition without sacrificing factuality or coverage.

**Cost:** Low to medium; no retraining, but beam search and longer outputs may increase runtime and memory.

**Validation:** EOS-before-limit and max-length rates, repeated trigram ratio, semantic repetition, empty/invalid outputs (define invalidity in advance), claim-level factuality, coverage, ROUGE and inference cost. A length-limit stop is not natural EOS; lexical anti-repetition is not factual verification.

## 6. Evaluation protocol and safeguards

- Preserve checkpoint-96, Config C, original holdout results and their reports as V1 records. The already examined 100-article holdout cannot serve as an independent test for a V2 selected using these findings.
- Establish a new development set and a separate untouched final evaluation set from eligible non-overlapping records before V2 selection. Confirm article identity, reference availability, dataset revision and near-duplicate risks. If independent data are unavailable, report this as a blocker rather than claiming a new independent evaluation.
- Fix the experimental hypotheses, candidate settings, metric definitions and any future release criteria before opening the new final set. Choose V2 on development data only; freeze the selected configuration before final inference.
- Evaluate identical articles across competing systems and retain all outputs, including empty and invalid ones. Use complete references, consistent ROUGE settings and paired bootstrap intervals for mean article-level ROUGE-L differences. Keep automatic and qualitative sample sizes explicit.
- Define key findings before inspecting model outputs. Distinguish Supported, Contradicted, Not Verifiable and Uninterpretable. Evaluate input-grounded and full-article factuality separately. Report counts and rates with explicit denominators, partial coverage and summary-level errors. Seek independent biomedical expert review and annotation agreement before making reliability claims.
- Record model/tokenizer revisions, adapter identity, configuration hash, data manifest, source selection version, seed, precision, training settings and resource measurements. Do not publicly commit raw articles or reference summaries; verify distribution rights for outputs, evidence excerpts and model artifacts.
- No production approval follows automatically from better ROUGE. A future release assessment must consider scientific contradiction severity, terminology, information coverage, output quality, expert validation and deployment risks. No new numerical release threshold is established by this V1 holdout review.

## 7. Suggested experiment order

| Priority | Experiment | Rationale |
|---|---|---|
| 1 | H1 — Source selection | Audit missing source information before adding training cost. |
| 2 | H4 — Decoding | Low-cost frozen-model ablation; isolate termination and repetition behavior. |
| 3 | H2 — Training data | Assess data scale and source-target alignment with controlled training. |
| 4 | H3 — Model capacity | Run larger-model comparisons after resource feasibility and data audits. |

This ordering is a proposed engineering decision based on cost and diagnostic value, not a demonstrated ranking of root causes. All experiments remain future work. No model or decoding configuration has been changed, and no additional holdout inference is performed by this plan.
