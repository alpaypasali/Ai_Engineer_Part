# NewsRoute — Model Card

## Model Overview
NewsRoute classifies English-language news text into four topic categories: **World (0)**, **Sports (1)**, **Business (2)**, and **Sci/Tech (3)**. It is a learning project for routing articles to the appropriate editorial team; it is not a production-approved decision system.

## Dataset and Training
- **Dataset:** AG News ([Hugging Face — fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news)).
- **Training:** 3,200 articles selected from the original AG News training split.
- **Validation:** 800 articles from that same selected sample, with 200 articles per class.
- **Final test:** The original AG News test split, containing 7,600 articles (1,900 per class), used only after model development and selection.

The original workflow sampled 1,000 articles per class (4,000 total), cleaned them, then created the 3,200/800 train/validation split. The corpus-frequency filter for words seen once was computed before the split; this is a **validation-methodology limitation**, not a recommended practice for future projects.

## Candidate Models
### TF-IDF + Logistic Regression — Selected
- `TfidfVectorizer(max_features=5000, ngram_range=(1, 2), lowercase=True)`
- `LogisticRegression(max_iter=1000)`
- Saved inference pipeline: `models/tfidf_pipeline.joblib`

### Mini Transformer — Comparison Model
- Trainable token embeddings and fixed sinusoidal positional encoding.
- One Transformer Encoder layer with four attention heads (`d_model=64`, `dim_feedforward=128`, `dropout=0.1`).
- Padding mask, masked mean pooling, and a four-class linear head.
- Trained from scratch for three epochs; no pretrained language model was used.
- Saved weights: `models/newsroute_transformer.pt`; vocabulary: `models/vocab.json`.

## Evaluation Results
All metrics were calculated on the same examples within each split.

| Model | Split | Samples | Accuracy | Macro Precision | Macro Recall | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF + LR | Validation | 800 | 0.8712 | 0.8719 | 0.8712 | **0.8711** |
| Mini Transformer | Validation | 800 | 0.7138 | 0.7143 | 0.7138 | 0.7108 |
| TF-IDF + LR | Final test | 7,600 | 0.8705 | 0.8699 | 0.8705 | **0.8701** |
| Mini Transformer | Final test | 7,600 | 0.7361 | 0.7374 | 0.7361 | 0.7360 |

The final test results reflect the original training-time text preprocessing. The test set was not used to select or retrain a model.

## Model Selection
**TF-IDF + Logistic Regression** was selected based on its higher **validation Macro-F1 (0.8711 vs 0.7108)**, stronger class-level recall, and simpler inference pipeline. The final test was used to report held-out performance, **not to change the model choice**.

## Input and Output
- **Input:** One non-empty English news string.
- **Output:** One of `World`, `Sports`, `Business`, or `Sci/Tech`.
- **CLI:** `python predict.py "NASA announces new satellite"`
- **Input validation:** Empty, whitespace-only, `None`, and numeric inputs are rejected by `predict_news()`.

Inference applies the saved baseline's preprocessing, including lowercasing, punctuation/number removal, stopword filtering, the training-derived `models/rare_words.json` list, and lemmatization. NLTK language resources must be available.

## Limitations and Risks
- Evaluated only on **English-language news** in four broad classes.
- The model was developed with a relatively small training sample (3,200 articles).
- Business and Sci/Tech stories, among others, can overlap; a wrong label may route a story to the wrong team.
- AG News is a historical dataset; behavior on current events, new language usage, and other publishers is not established.
- The original preprocessing's rare-word filter was derived before train/validation splitting, which can make validation estimates less independent.
- The model does not offer reliable uncertainty calibration or an approved human-review workflow.
- Real-world reliability, security, monitoring, and performance on contemporary news would need separate validation before production use.

## Intended Use
Demonstration, education, and prototyping of news-topic routing. Predictions should be treated as suggestions, not as authoritative decisions in a live editorial workflow.

## Reproducibility and Tests
- Model artifacts: `models/tfidf_pipeline.joblib`, `models/rare_words.json`.
- Run demo: `python predict.py "Football team wins championship"`.
- Run automated tests: `python -m pytest -q`.
- Six tests passed in the user's local environment: empty input, whitespace-only input, missing model file, `None`, integer input, and valid news input. No fully fresh environment was created for this exercise.
