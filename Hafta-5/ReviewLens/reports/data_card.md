# ReviewLens — Data Card

## Dataset

| Property | Description |
|---|---|
| Name | Yelp Review Full |
| Source | [Hugging Face dataset card](https://huggingface.co/datasets/Yelp/yelp_review_full) |
| Original source | Yelp Dataset Challenge 2015 |
| Language | Primarily English |
| Task | Star rating classification; experimental semantic retrieval |
| Original splits | 650,000 train / 50,000 test |
| Original fields | `text`, `label` |
| Local derived field | `stars = label + 1` |

## Local working sample

The project used 1,000 reviews selected from the original train split with equal representation of all five classes.

| Stars | Local sample | Classification evaluation |
|---|---:|---:|
| 1 | 200 | 40 |
| 2 | 200 | 40 |
| 3 | 200 | 40 |
| 4 | 200 | 40 |
| 5 | 200 | 40 |
| Total | 1,000 | 200 |

Local fields: `text` (review), `label` (0–4), `stars` (1–5). A local row index identifies each sample.

## Sampling and evaluation

- Retrieval used a corpus of 800 reviews and 200 queries. Corpus and query row indices and exact texts were checked for overlap.
- Classification used a fixed 200-review stratified evaluation sample (`test_size=0.20`, `random_state=42`). The saved evaluation indices preserved order for both candidates.
- Model A and Model B used identical evaluation records and labels. No candidate was fine-tuned.
- Latency was measured on 20 evaluation reviews, four from each rating.
- The original dataset test split was not used. Some reviews may have appeared in earlier exploratory work. These measurements are development results.

## Preprocessing

The original review text was used as model input. Both classification candidates used truncation at 256 tokens and batch size 8. Their tokenizers differ, so the retained text may differ. The embedding model produces normalized 384-dimensional representations with a 256-token limit.

## Limitations and bias

The balanced sample is useful for comparing classes but is not necessarily representative of the natural frequency of ratings or current review distributions. Yelp reviews cover different business types and should not be treated as a restaurant-only population unless filtered and validated. Ratings are subjective and are not semantic relevance labels. Long reviews can lose context through truncation. Text may also contain personal information.

## Data rights and publication

The dataset card identifies the license as `other` and points to the [Yelp Dataset Terms of Use](https://s3-media3.fl.yelpcdn.com/assets/srv0/engineering_pages/bea5c1e92bf3/assets/vendor/yelp-dataset-agreement.pdf).

The agreement describes academic/non-commercial use and restrictions on redistributing data and derived metrics. Sections 3 and 4 require review of publication conditions; obtain Yelp's required approval before publishing findings involving the dataset or brand. Model licenses are separate.

Keep sample CSVs, retrieval corpora, row-level prediction exports, evaluation indices and review-containing notebook outputs private. Do not distribute embeddings or other derivatives without confirmed authorization. This report and aggregate results are a private educational draft pending any required publication approval. Synthetic test examples contain no dataset records.

## Citation

Zhang, X., Zhao, J., & LeCun, Y. (2015). [Character-level Convolutional Networks for Text Classification](https://arxiv.org/abs/1509.01626). Advances in Neural Information Processing Systems.
