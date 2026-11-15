# ReviewLens — Model Card

## Model identity and intended use

| Property | Selected classifier | Retrieval encoder |
|---|---|---|
| Model ID | `nlptown/bert-base-multilingual-uncased-sentiment` | `sentence-transformers/all-MiniLM-L6-v2` |
| Architecture | Multilingual BERT | MiniLM-L6 |
| Task | Five-class sentiment rating | Semantic text similarity |
| Output | 1–5 stars + confidence | 384-dimensional dense vector |
| License (model card) | MIT | Apache-2.0 |
| Fine-tuning in ReviewLens | None | None |
| Maximum input length | 256 tokens | 256 wordpieces |

NLP Town BERT was fine-tuned on product reviews in English, Dutch, German, French, Spanish and Italian. MiniLM is an English sentence and short-paragraph encoder. ReviewLens evaluation was performed on Yelp reviews in English; multilingual accuracy was not established.

## Input and output

- Input: non-empty review text or a non-empty batch.
- Classification: `1 star` through `5 stars` are explicitly mapped to integers 1–5. Invalid labels are rejected.
- Confidence: model score for the predicted class, between 0 and 1; not a calibrated guarantee of correctness.
- Retrieval: normalized embedding dot product yields cosine similarity. Return up to five corpus rows, ordered by decreasing similarity.

## Evaluation and selection

On the same 200 balanced development examples:

| Metric | Selected BERT | Alternative DistilBERT |
|---|---:|---:|
| Accuracy | 0.5550 | 0.4200 |
| Macro-F1 | 0.5532 | 0.3827 |
| 1-Star Recall | 0.6000 | 0.5000 |
| 1-Star Error Rate | 40% | 50% |
| Mean latency (ms) | 16.76 | 7.16 |
| P95 latency (ms) | 32.26 | 14.61 |

Model A correctly classified 24 of 40 actual 1-star reviews. Model B (TabularisAI) correctly classified 20 of 40. Model B is faster, but it predicts five sentiment categories, which require an approximate star mapping. Its model card also lists a non-commercial license (CC BY-NC 4.0).

Mean and P95 timings are exploratory results from 20 single-review GPU measurements, not guaranteed production latency.

The chosen model better meets the project's goal of detecting low-star feedback. Macro-F1 weights the F1 of every class equally. The 1-star error metric additionally tracks missed critical negative feedback.

## Limitations and risks

- Product-review training is not equivalent to Yelp-specific adaptation.
- Middle ratings are difficult to distinguish. In the previous evaluation, 3-star recall was 0.375.
- Long inputs can lose important suffixes. An actual synthetic test shortened 895 tokens to 256 and missed a serious negative statement at the end.
- A synthetic sarcastic complaint received 5 stars with confidence 0.8792.
- A mixed positive/negative review received 3 stars with confidence 0.5088; no true rating was supplied for that text.
- The retrieval model measures semantic relevance, not sentiment agreement. Its same-star comparison is only a proxy.
- The 200-example development set is not an independent final test. Model versions were not pinned by immutable Hub revision during the original experiments.

## Appropriate use

Use for educational experimentation and as a prototype for review analysis. Do not automatically suppress complaints, assign authoritative customer ratings, or use the model as the sole decision-maker. Human review, independent evaluation, calibration, monitoring and lawful data access are needed before deployment.

The MIT classifier license and Apache-2.0 encoder license are model-specific. They do not provide permission to publish or commercially use Yelp data. Experimental metrics in this card are a private educational draft until required publication approval is obtained.

## Sources

- [NLP Town BERT model card](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
- [MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
- [TabularisAI model card](https://huggingface.co/tabularisai/multilingual-sentiment-analysis)
- [ReviewLens data card](data_card.md)
