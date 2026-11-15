# ReviewLens — Error Scenarios and Improvements

## Scenario results

The following results were measured with the actual pretrained NLP Town BERT on CPU, with `max_length=256`, truncation enabled and no additional training. The examples are synthetic and not Yelp dataset records.

| Scenario | Input / description | Actual result | Observation |
|---|---|---|---|
| Empty input | Empty string and whitespace | ValueError | Invalid input rejected before inference. |
| Long review | Positive text repeated 80 times, followed by a serious negative sentence | 5 stars, score 0.7682 | Token count: 895 → 256. The final negative statement was truncated. |
| Batch | Amazing food. / Terrible service. / The food was average. | 5 / 1 / 3 stars; scores 0.8730 / 0.8419 / 0.5412 | Three ordered outputs for three inputs. |
| Irony | Great, another forty minutes waiting for cold food. Just perfect. | 5 stars, score 0.8792 | A negative experience was predicted as very positive. High confidence does not guarantee correctness. |
| Mixed sentiment | The food was excellent, but the staff were rude and the service was slow. | 3 stars, score 0.5088 | A middle rating is plausible; there is no labeled ground truth for this synthetic text. |

## Reproduced and fixed problems

1. Empty input raised a Python traceback in the command-line interface. Added explicit CLI input validation using `parser.error`, so the user now receives an understandable error message and a nonzero exit code without a traceback. The original Python API still raises `ValueError`.
2. A corpus containing fewer than five reviews was rejected. The requirement is at most five results, not always five. The corpus must now contain at least one valid review, and search returns `min(5, corpus_size)` distinct rows. An empty corpus is still rejected.

Two regression tests failed before these fixes and passed afterward. Irony and truncated suffixes remain model limitations; the fixes do not claim to solve these issues or improve Macro-F1.

## Automated tests

Run `python -m unittest discover -s tests -v` from the project root. Ten tests cover label mapping, single prediction, star/confidence range, batch order and size, invalid input, friendly CLI validation, empty corpus, small corpus retrieval, ordered Top-5 retrieval and search without corpus.

Run `python -m tests.error_scenarios` to reproduce the five scenarios with the actual models and verify small-corpus retrieval. Run `python -m tests.smoke_test` for the original end-to-end test. All model tests use synthetic text.

## What I learned

Input validation and small-corpus handling are fixable implementation problems. Irony, mixed sentiment and lost context after truncation are model limitations. A valid score and a successful software test do not imply that a predicted rating is correct.
