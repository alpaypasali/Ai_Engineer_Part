# Day 03 — Embedding & Semantic Retrieval

## 1. Problem

In ReviewLens, I wanted to find reviews that are similar to a given Yelp review. I compared two different text representations: TF-IDF and pretrained sentence embeddings.

## 2. Dataset

I used the Yelp Review Full dataset. Each review includes text and a star rating from 1 to 5.

## 3. Corpus / Query Split

- Corpus: 800 reviews
- Query: 200 reviews

The corpus was used for searching, while the queries were used to test retrieval results.

## 4. TF-IDF Representation

TF-IDF represents text using word frequency and importance. It produces sparse vectors and mainly relies on shared words.

The disadvantage is that it cannot directly understand semantic relationships between different words.

## 5. Pretrained Embedding Model

I used `sentence-transformers/all-MiniLM-L6-v2`.

The model was used with pretrained weights. No additional training or fine-tuning was performed.

## 6. Dense Embeddings

The model converts each review into a 384-dimensional dense vector.

- Corpus embeddings: (800, 384)
- Query embeddings: (200, 384)

I used `normalize_embeddings=True` to normalize the vectors.

## 7. Cosine Similarity

I calculated cosine similarity for both representations.

- TF-IDF similarity matrix: (200, 800)
- Embedding similarity matrix: (200, 800)

For normalized embeddings, dot product is equivalent to cosine similarity.

## 8. Top-1 and Top-5 Retrieval

For each query, I searched the corpus and retrieved the five reviews with the highest similarity scores.

I used NumPy sorting operations to obtain the Top-1 and Top-5 results.

## 9. Model Comparison

| Metric | TF-IDF | Embedding |
|---|---:|---:|
| Corpus Samples | 800 | 800 |
| Query Samples | 200 | 200 |
| Top-1 Same-Star Rate | 0.335 | 0.305 |
| Top-5 Same-Star Fraction | 0.291 | 0.297 |

TF-IDF performed slightly better on Top-1 Same-Star Rate.

Embedding performed slightly better on Top-5 Same-Star Fraction.

However, these metrics do not directly measure semantic retrieval quality.

## 10. Qualitative Analysis

I manually compared five queries where the two methods returned different Top-1 results.

| Query ID | TF-IDF Top-1 ID | Embedding Top-1 ID | Observation |
|---|---:|---:|---|
| 0 | 246 | 385 | Embedding was more relevant to the bar and beer topic. |
| 1 | 680 | 791 | Embedding found a more relevant fast-food service complaint. |
| 2 | 742 | 744 | TF-IDF better matched the seating and reception problem. |
| 3 | 716 | 450 | TF-IDF better matched the disappointing restaurant experience. |
| 6 | 749 | 332 | Embedding better matched the negative barista interaction. |

Embedding appeared more relevant in three examples, while TF-IDF was more relevant in two.

Five examples are not enough to determine the overall winner.

## 11. Limitations

### Same-Star Proxy

Reviews with the same star rating are not necessarily semantically similar.

Same-Star Rate is only a proxy metric, not a direct measurement of semantic retrieval accuracy.

### Long Reviews

The MiniLM model uses a default maximum input length of 256 WordPiece tokens.

Longer reviews may be truncated, causing important information to be lost.

### Dataset Size

The experiment used only 800 corpus reviews and 200 queries.

The results may change with a larger dataset.

### Evaluation

I did not use human-labeled semantic relevance judgments.

A stronger evaluation would require relevance labels.

## 12. Automated Tests

All three automated tests passed.

| Test | Result |
|---|---|
| Corpus / Query Split | PASSED |
| Representation Shapes | PASSED |
| Top-5 Retrieval | PASSED |

The tests checked dataset separation, representation dimensions, finite values, and retrieval index validity.

## 13. What I Learned

In this experiment, I learned that TF-IDF and embeddings represent text differently.

TF-IDF focuses on word statistics, while embeddings can capture semantic relationships.

I also learned how cosine similarity, normalization, and Top-K retrieval work.

One important finding was that having the same star rating does not mean two reviews are semantically similar.

Embedding was not always better than TF-IDF. The results depended on what I measured and which reviews I examined.

I also understood that retrieval and classification are different tasks. Retrieval finds similar reviews, while classification predicts a category.

## 14. Data Sharing

Raw Yelp reviews and notebook outputs containing review text are not included in this report.

Before publishing CSV files or other derived artifacts, the dataset's redistribution terms must be checked.

Data without verified redistribution permission should remain private.
