# HealthBrief — Day 02: PEFT and LoRA Preparation

**Date:** 2026-11-24  
**Project:** HealthBrief — Biomedical Lay Summarization  
**Notebook:** `notebooks/03_peft_lora_preparation.ipynb`  
**Base model:** `google/flan-t5-small`  
**Status:** Preparation and adapter configuration completed; training not started. Standalone test execution remains unverified.

## 1. Objective

Prepare a reproducible parameter-efficient fine-tuning (PEFT) experiment for biomedical lay summarization. The task is to adapt FLAN-T5-small to generate accessible summaries of research articles while keeping the original model weights frozen.

Day 02 focuses on source selection, leakage-controlled dataset preparation, model architecture inspection, theoretical memory analysis, LoRA configuration, tokenization, dynamic padding, and training planning. No optimizer updates, adapter training, or final holdout evaluation were performed.

## 2. Dataset

The experiment uses `BioLaySumm/BioLaySumm2025-eLife`, a biomedical article and lay-summary dataset. Relevant fields include `article`, `summary`, and `section_headings`.

| Property | Value |
|---|---|
| Original train split | 4,346 articles |
| Original validation split | 241 articles |
| Original test split | 142 articles |
| Dataset revision used for holdout retrieval | `144e9c785e3a309da804eb6786feb38e6596f390` |
| Sampling seed | 42 |
| Reference language | English |

The existing Day 01 sampled dataset was retained. The original validation split was explicitly reloaded at the recorded revision to construct an independent reference-bearing holdout. The initial Day 01 dataset load was not revision-pinned; the selected validation article identities were subsequently checked against the pinned validation split. Full revision equivalence of all original sampled records has not been established.

Day 01 analysis showed that source articles are much longer than the planned encoder input. Training article median length was 16,272 tokens, compared with a median reference length of 519 tokens. Direct truncation to 512 tokens discarded approximately 97% of source tokens on average, motivating an explicit source-selection strategy.

## 3. Source Selection

### Strategy (V1)

`prepare_source_text(article, section_headings, tokenizer, max_length)` attempts to identify the abstract and a conclusion/discussion/results section using section metadata and recognizable heading patterns. When reliable headings cannot be found, it uses the opening text and a candidate paragraph selected through result-oriented lexical cues (e.g., `we found`, `our results`, `these findings`). It does not use the reference summary to choose source passages.

The selected passages are combined with the instruction:

```text
Summarize the following biomedical research article in plain language for a general audience:
```

The total encoder input budget is 512 tokens, including instruction and special tokens. In the initial inspection, the prompt and special tokens accounted for 20 tokens, leaving approximately 492 tokens for source text. The function allocates the available budget across the selected passages, redistributes unused capacity, and checks the final input length.

### Observed findings

- Ten training examples were inspected in the initial source-selection analysis. All ten used the opening/result-cue fallback; no headings were verified by the implemented detection logic.
- These examples retained 492 source tokens each, with an average source retention rate of approximately 3.36%.
- Manual inspection of three examples showed that useful content could be retained, but sentence boundaries could be cut and a result-cue paragraph could still contain introductory or methodological material rather than the actual findings.
- The final tokenized training and validation inputs ranged from 511 to 512 tokens.

V1 was retained for the initial LoRA experiment. Its token-budget compliance is not evidence that all important facts were preserved. Section-level semantic accuracy and summary factual coverage remain open issues.

**Output:** `reports/source_selection_analysis.csv`.

## 4. Dataset Preparation

### Fixed split design

| Split | Records | Reference summaries | Intended use |
|---|---:|---:|---|
| Train | 500 | 500 | Adapter training |
| Validation | 100 | 100 | Development evaluation and checkpoint selection |
| Final holdout | 100 | 100 | End-of-week evaluation only |
| Original test (sampled) | 100 | 0 | Unlabeled test / inference only |
| **Total** | **800** | **700** | |

Article IDs were generated using SHA-256 over lowercase, whitespace-normalized article text. The existing 100 validation identities and training identities were excluded from the pinned original validation pool. The remaining 141 reference-bearing candidates were sampled without replacement using `random_state=42` to construct the 100-record holdout.

Observed train–validation, train–holdout and validation–holdout intersections were all zero. This is an exact normalized-text identity check, not a near-duplicate or semantic leakage audit. The sampled original test split was not used for training or model selection.

The split manifest records each article ID, split name, and reference availability. Source selection was applied only to train and validation. Each prepared record includes `split`, `article_id`, `input_text`, `target_summary`, `source_token_count`, `target_token_count`, and `source_sections`.

### Target truncation analysis

| Split | Target limit | Truncated records | Truncation rate (%) | Mean per-example token loss (%) |
|---|---:|---:|---:|---:|
| Train | 256 | 500/500 | 100 | 49.180867 |
| Train | 512 | 260/500 | 52 | 6.740734 |
| Train | 768 | 4/500 | 0.8 | 0.030313 |
| Validation | 256 | 100/100 | 100 | 48.809394 |
| Validation | 512 | 48/100 | 48 | 7.362712 |
| Validation | 768 | 0/100 | 0 | 0 |

Truncation rate measures the proportion of summaries exceeding the limit. Mean token loss is the average of individual lost-token fractions, not a direct measurement of semantic information loss.

**Decision:** The original Day 03 plan specified a target limit of 256. It was revised to **768** because 256 would truncate every reference and remove nearly half of each summary's tokens on average. A limit of 512 remains a possible fallback if memory is insufficient; it would require a new tokenization run and configuration update.

### Data artifacts

| Artifact | Purpose |
|---|---|
| `data/healthbrief_sampled/` | Existing sampled raw data |
| `data/final_holdout/` | Separate reference-bearing holdout |
| `data/healthbrief_prepared/` | Prepared train/validation text pairs |
| `data/healthbrief_tokenized/` | Tokenized train/validation data; save/reload verification pending evidence |
| `reports/dataset_split_manifest.csv` | Stable split identities |
| `reports/training_data_quality.csv` | Target truncation analysis |

## 5. Model Architecture

The base model was loaded with `AutoModelForSeq2SeqLM`. The following values were obtained from its configuration and parameter tensors before LoRA injection.

| Feature | Observed value |
|---|---|
| Architecture | `T5ForConditionalGeneration` |
| Encoder blocks | 8 |
| Decoder blocks | 8 |
| Hidden size (`d_model`) | 512 |
| Attention heads | 6 |
| Head dimension (`d_kv`) | 64 |
| Attention inner dimension | 384 |
| Feed-forward intermediate size | 1,024 |
| Feed-forward type | `gated-gelu` |
| Vocabulary capacity | 32,128 |
| Base model parameters | 76,961,152 |
| Initially trainable parameters | 76,961,152 |
| Initially frozen parameters | 0 |
| Loaded parameter dtype | `torch.float32` (first parameter inspected) |
| Parameter tensor memory | 293.583496 MiB |

The encoder uses self-attention and a feed-forward network. The decoder uses causal self-attention, encoder–decoder cross-attention, and a feed-forward network. Cross-attention queries come from decoder representations; keys and values come from encoder representations.

### Actual module names and shapes

| Component | Example module path (first block) |
|---|---|
| Encoder self-attention | `encoder.block.0.layer.0.SelfAttention` |
| Encoder feed-forward | `encoder.block.0.layer.1.DenseReluDense` |
| Decoder self-attention | `decoder.block.0.layer.0.SelfAttention` |
| Decoder cross-attention | `decoder.block.0.layer.1.EncDecAttention` |
| Decoder feed-forward | `decoder.block.0.layer.2.DenseReluDense` |

| Projection | Weight shape (out, in) | Weight parameters |
|---|---|---:|
| `q`, `k`, `v` (each) | (384, 512) | 196,608 |
| `o` | (512, 384) | 196,608 |
| `wi_0`, `wi_1` (each) | (1,024, 512) | 524,288 |
| `wo` | (512, 1,024) | 524,288 |

Despite the historical `DenseReluDense` module name, the inspected model uses `T5DenseGatedActDense` with `NewGELUActivation`. The projection table was inspected for the first encoder and decoder blocks (18 linear modules), not every linear layer in the model.

**Outputs:** `reports/model_architecture.csv`, `reports/model_projection_analysis.csv` (auxiliary first-block analysis).

## 6. Model Memory

The base model has 76,961,152 parameters. Actual parameter tensor bytes were computed as the sum of `p.numel() * p.element_size()` for each parameter. Theoretical uniform-precision memory was calculated as:

```text
Weight bytes = parameter count × (bits per parameter / 8)
Weight MiB   = weight bytes / 1024²
```

| Precision | Bits | Bytes/parameter | Theoretical weight memory (MiB) | Reduction vs. FP32 (%) |
|---|---:|---:|---:|---:|
| FP32 | 32 | 4 | 293.583496 | 0 |
| FP16 | 16 | 2 | 146.791748 | 50 |
| BF16 | 16 | 2 | 146.791748 | 50 |
| INT8 | 8 | 1 | 73.395874 | 75 |
| INT4 | 4 | 0.5 | 36.697937 | 87.5 |

These figures are weight-only calculations, not measured GPU memory. Actual training also requires gradients, optimizer states, activations, temporary tensors, buffers, and allocator overhead. Quantized weights may require scale factors and other metadata, and some layers may remain unquantized.

For illustration, assuming all relevant tensors are FP32 and Adam has two FP32 moment tensors per trainable parameter, full fine-tuning requires approximately 1,174.33 MiB for weights, gradients and moments combined. Rank-8 LoRA requires approximately 298.83 MiB for the same categories, including base and adapter weights. Both estimates exclude activations and other overhead; neither is a measured peak.

**Output:** `reports/model_memory_analysis.csv`.

## 7. PEFT

Parameter-efficient fine-tuning adapts a pretrained model while updating only a small subset of parameters or newly added trainable parameters. LoRA represents the learned change to a frozen linear weight matrix as a low-rank product:

```text
W ∈ R^(d_out × d_in)
A ∈ R^(r × d_in)
B ∈ R^(d_out × r)
ΔW = B A
W_effective = W + (alpha / r) B A
P_adapter = r × (d_in + d_out)
```

For a hypothetical 768 × 768 weight matrix (589,824 original weights, excluding bias):

| Rank | A shape | B shape | Adapter parameters | Reduction in trainable parameters (%) |
|---:|---|---|---:|---:|
| 4 | (4, 768) | (768, 4) | 6,144 | 98.96 |
| 8 | (8, 768) | (768, 8) | 12,288 | 97.92 |
| 16 | (16, 768) | (768, 16) | 24,576 | 95.83 |
| 32 | (32, 768) | (768, 32) | 49,152 | 91.67 |

Increasing rank increases adapter capacity and parameter count, but does not guarantee higher ROUGE or factual consistency. This hypothetical layer is separate from the actual FLAN-T5 projection dimensions.

## 8. LoRA Configuration

PEFT `LoraConfig` and `get_peft_model()` were used to inject adapters into the base model. The PEFT installation output reported version `0.22.0`.

| Parameter | Value |
|---|---|
| Task type | `SEQ_2_SEQ_LM` |
| Rank | 8 |
| Alpha | 16 |
| Standard scaling (`alpha/r`) | 2 |
| LoRA dropout | 0.05 |
| Target modules | `q`, `v` |
| Bias | `none` |
| Inference mode | `False` |

The selected projections cover encoder self-attention, decoder self-attention, and decoder cross-attention. Query adapters modify query representations used in attention scoring; value adapters modify the content representations transferred through attention. Key, output, feed-forward, and embedding weights remain frozen.

Standard LoRA initialization starts with a zero effective update (B is initialized to zero), preserving the base model's initial mapping. No optimization has occurred yet.

## 9. Parameter Analysis

### Observed model comparison

| Metric | Full fine-tuning (base model, pre-training) | LoRA r=8 (configured) |
|---|---:|---:|
| Total parameters | 76,961,152 | 77,305,216 |
| Trainable parameters | 76,961,152 | 344,064 |
| Frozen parameters | 0 | 76,961,152 |
| Trainable fraction of respective model (%) | 100 | 0.445072 |

Relative to full fine-tuning, the number of trainable parameters is reduced by approximately 99.553%. The total parameter count increases because LoRA adds new tensors without deleting original weights. This is a parameter comparison, not a trained performance or runtime comparison.

### Theoretical rank comparison on actual q/v projections

| Rank | Target locations | Adapter parameters | Total after injection | Trainable ratio (%) |
|---:|---:|---:|---:|---:|
| 8 | 48 | 344,064 | 77,305,216 | 0.445072 |
| 16 | 48 | 688,128 | 77,649,280 | 0.886200 |

Only rank 8 was instantiated. The rank-16 row remains theoretical.

### Observed adapter distribution

| Component | Adapter locations | Trainable tensors | Trainable parameters |
|---|---:|---:|---:|
| Encoder self-attention | 16 | 32 | 114,688 |
| Decoder self-attention | 16 | 32 | 114,688 |
| Decoder cross-attention | 16 | 32 | 114,688 |
| **Total** | **48** | **96** | **344,064** |

Each adapted projection contains A (8, 512) with 4,096 parameters and B (384, 8) with 3,072 parameters. Across 48 adapted projections, these sum to 344,064. The observed model matches the theoretical rank-8 count exactly.

**Outputs:** `reports/lora_parameter_analysis.csv`, `reports/lora_theoretical_analysis.csv`, `reports/lora_adapter_details.csv`, `reports/lora_adapter_distribution.csv`.

## 10. Quantization

Quantization changes how weights are represented, not the logical number of model weights. INT8 uses 8-bit integer representations; NF4 uses 4-bit code levels designed for approximately normally distributed weights. Both can introduce approximation error and change predictions.

FP16 and BF16 both use 16 bits. FP16 has five exponent bits and ten fraction bits, while BF16 has eight exponent bits and seven fraction bits. BF16 therefore provides a wider dynamic range but less local numerical precision than FP16, often reducing overflow risk in neural network training.

QLoRA combines a frozen 4-bit quantized base model with trainable LoRA adapters. Storage precision and computation precision are distinct: a base weight can be stored in NF4 while matrix computations use BF16. Standard quantized PEFT workflows train additional parameters rather than directly updating the quantized base weights.

### Observed hardware environment

| Feature | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| Reported GPU memory | 7.624573 GiB |
| CUDA available | True |
| Compute capability | 8.9 |
| PyTorch version (reported) | 2.14.1+cu130 |
| PyTorch CUDA build (reported) | 13.0 |
| Native BF16 support | True |

**Decision:** Use standard LoRA with BF16 mixed precision for the initial experiment. A 4-bit or 8-bit loading experiment was deliberately skipped: the base model is small, and quantization would introduce additional complexity and approximation risk. Actual bitsandbytes compatibility was not tested. No quantized checkpoint, inference comparison, or memory measurement was produced.

## 11. Training Configuration

### Planned Day 03 experiment

| Parameter | Value |
|---|---|
| Base model | `google/flan-t5-small` |
| Fine-tuning | LoRA, r=8, alpha=16, dropout=0.05 |
| Target modules | `q`, `v` |
| Epochs | 3 |
| Learning rate | 2e-4 |
| Train micro batch size | 4 |
| Evaluation batch size | 4 |
| Gradient accumulation steps | 4 |
| GPU count | 1 |
| Effective full accumulation batch | 16 examples |
| Maximum source length | 512 tokens |
| Maximum target length | 768 tokens (revised from 256) |
| BF16 mixed precision | Enabled |
| FP16 | Disabled |
| Quantization | Disabled |
| Padding | Dynamic |
| Label padding | -100 |
| Seed | 42 |
| Evaluation | ROUGE |
| Checkpoint selection | Highest validation ROUGE-L |

These are starting settings, not optimized hyperparameters. The generation length limit and decoding configuration for the training evaluation still need to be explicitly recorded on Day 03. A target token limit is not automatically the same as a generation token limit. BF16 and batch size 4 have not been validated with a training forward/backward memory measurement.

### Tokenization and collator

Train and validation were separately processed with the same tokenizer. Source and target were tokenized independently with truncation and no static padding; target token IDs were assigned to `labels`. The resulting dataset contains `input_ids`, `attention_mask`, and `labels`.

| Token length statistic | Train | Validation |
|---|---:|---:|
| Source minimum | 511 | 511 |
| Source maximum | 512 | 512 |
| Source mean | 511.898 | 511.89 |
| Target minimum | 258 | 302 |
| Target maximum | 768 | 759 |
| Target mean | 521.33 | 521.41 |

`DataCollatorForSeq2Seq` was instantiated using the tokenizer and LoRA model. The first four training examples had target lengths 466, 349, 346 and 625. Dynamic padding produced `(4, 512)` encoder tensors and `(4, 625)` label/decoder-input tensors. A total of 714 padded label positions were assigned `-100`, leaving 1,786 non-padding label positions for the loss calculation.

Teacher forcing uses shifted reference tokens as decoder inputs. The decoder start token was 0 and shifted `-100` values were replaced by the model's pad token. Label padding is excluded from cross-entropy loss, but does not eliminate all associated computation or memory cost.

### Expected training steps

| Metric | Per epoch | Three epochs |
|---|---:|---:|
| Training examples processed | 500 | 1,500 |
| Micro batches | 125 | 375 |
| Optimizer update steps | 32 | 96 |

The final accumulation group contains only four examples per epoch. These estimates assume one GPU, no dropped batches, and an optimizer update for the incomplete accumulation group at the end of each epoch. Actual steps will be checked against trainer logs. Gradient accumulation reduces peak activation demand compared with processing 16 examples simultaneously, but exact numerical equivalence is not guaranteed due to dropout, padding and token-level loss normalization.

**Output:** `reports/training_configuration.csv`.

## 12. Limitations

- **Source coverage:** The encoder budget retains only a small fraction of each article. V1 can miss decisive findings or choose the wrong passage, and token-boundary truncation can leave incomplete sentences.
- **Source-target mismatch:** A complete lay summary may include facts not present in the selected source. This may encourage unsupported generation even when training loss improves.
- **Evaluation limitations:** ROUGE measures lexical overlap and cannot guarantee scientific accuracy, readability, or factual consistency. Manual error analysis remains necessary. ROUGE for the final holdout should use the original reference summaries and not drive hyperparameter selection.
- **Data scale and leakage:** Only 500 training samples are used. Hash-based checks do not identify paraphrased duplicates, article variants or semantic overlap. Initial raw sampling was not revision-pinned in Day 01.
- **Model and hyperparameters:** A small base model and rank-8 adapters may have limited capacity. Rank, dropout, epochs and learning rate have not been tuned or compared through training.
- **Memory and hardware:** Weight-only calculations are not peak GPU measurements. BF16 mixed precision and 768-token targets are planned, not proven feasible with batch size 4. Quantized loading was not tested.
- **Reproducibility:** The dataset revision is recorded, but the model/tokenizer revision has not been pinned in the demonstrated workflow. The training seed should be set before creating fresh adapters and the complete software environment should be recorded for Day 03.
- **Distribution:** The dataset's redistribution license was not confirmed in Day 01. Raw or derived dataset artifacts should not be pushed to a public repository without verifying permission.
- **Baseline failure:** Day 01 reported an empty summary at validation index 54. The error must remain visible and must not be replaced with a fallback summary or silently removed from evaluation.

## 13. Tests

### Verified notebook checks

| Check | Evidence / status |
|---|---|
| Split isolation (train/validation/holdout) | PASS — all three observed intersections were zero |
| Manifest counts and reference availability | PASS — 800 records, 700 references |
| Manual LoRA parameter assertions | PASS |
| Adapter structure validation | PASS |
| LoRA matrix validation | PASS |
| Encoder–decoder distribution | PASS |
| Frozen parameter validation | PASS |
| Theoretical parameter consistency | PASS |
| Single-example tokenization equivalence | PASS |
| Full tokenized split counts and length statistics | Observed — 500/100 records within planned length limits |
| Four-example collator and padding inspection | Observed — expected shapes, right shift and 714 ignored label positions |
| Full dataset integrity assertion cell and tokenized save/reload assertion cell | Execution output not provided; status unverified |

### Standalone pytest suite

`tests/test_lora_preparation.py` contains eight structural/preparation tests and one baseline-output validity regression test. The file passed a Python syntax compilation check. No actual pytest execution result has been provided from the project environment.

| Test | Purpose | Runtime status |
|---|---|---|
| `test_dataset_integrity` | Split counts, normalized hash isolation, manifest identity/order | Pending |
| `test_source_selection` | Prompt and 512-token input limit across prepared records | Pending |
| `test_source_provenance` | Selected normalized text fragments found in original article; reference preserved | Pending |
| `test_model_architecture` | T5 encoder–decoder architecture and dimensions | Pending |
| `test_lora_configuration` | Rank, alpha, dropout, task, bias and target modules | Pending |
| `test_frozen_parameters` | Non-adapter parameters frozen | Pending |
| `test_trainable_parameters` | Exact expected adapter names, shapes, counts | Pending |
| `test_tokenization` | Retokenization consistency, valid IDs, collator shapes and padding | Pending |
| `test_baseline_sample_54_output_validity` | Detect/report empty output using the unadapted baseline model | Pending |

Run from the HealthBrief project root:

```bash
python -m pytest tests/test_lora_preparation.py -v
```

The regression test also checks that the summary validator rejects empty, whitespace-only and null values. It regenerates sample 54 with the unadapted model, the original source prefix (512 input tokens), greedy decoding and a 512-new-token cap. It writes `reports/baseline_sample_54_validation.json` with generated IDs, decoded output, and validity status before asserting that the summary is nonempty.

If the model still emits an empty summary, the regression test is expected to fail explicitly with `INVALID_SUMMARY`; this is evidence that the failure is detected, not that the model has been repaired. A nonempty output would pass the nonemptiness test but would not establish factual accuracy. The regeneration outcome is currently unknown. The source-provenance test checks textual containment, not the semantic correctness of detected section labels.

## 14. Conclusion and Next Steps

Day 02 established a fixed split manifest, a source-selection and tokenization pipeline, a reproducible starting LoRA configuration, and a theoretical memory comparison. The measured rank-8 parameter count matched the predicted 344,064 trainable parameters across 48 adapter locations. All five final notebook LoRA configuration test groups passed.

Before starting Day 03 training, verify the saved tokenized dataset and execute the standalone tests, recording any failures. Recreate the base model and LoRA configuration with the training seed set before adapter initialization, check the actual GPU memory requirements, and explicitly define generation settings and validation ROUGE-L checkpoint selection. Keep the final holdout untouched until the final evaluation.

The experiment is prepared, not trained. No improvement in ROUGE, factual consistency, runtime, or actual memory consumption is claimed.
