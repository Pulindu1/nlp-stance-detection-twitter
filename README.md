# nlp-stance-detection-twitter

Stance detection on Twitter rumour threads — lexical analysis, a fine-tuned
transformer classifier, instruction-prompted generation, and a two-stage
cascade, evaluated on **RumourEval 2017 (SemEval-2017 Task 8, Subtask A)**.

Given a source tweet making a claim and a reply to it, the task is to classify
the reply's stance toward the claim as **S**upport, **D**eny, **Q**uery or
**C**omment (SDQC). The dominant difficulty is class imbalance: `Comment`
accounts for ~64% of training replies, while `Deny` and `Query` together
account for ~15%.

## Approaches

| Notebook | Approach | Model |
|---|---|---|
| `1.ipynb` | Corpus analysis — tokenisation, stopword filtering, unigram/bigram frequency profiles and token distributions per stance class | — |
| `2a.ipynb` | Direct 4-way classification with layer-wise learning-rate decay | `vinai/bertweet-base` |
| `2b.ipynb` | Zero-shot and few-shot prompting, scored by ranked label likelihood, with per-class logit calibration | `google/flan-t5-large` |
| `2c.ipynb` | Two-stage cascade: binary comment detection, then S/D/Q classification on routed examples, with a tuned routing threshold | `vinai/bertweet-base` ×2 |

Each transformer run uses weighted cross-entropy against the class imbalance,
a fixed seed, and early stopping on dev macro-F1.

## Results (development set)

Macro-F1 is the headline metric — accuracy is dominated by the `Comment` class,
so a majority-class predictor already reaches ~0.65 accuracy at 0.20 macro-F1.

| Approach | Accuracy | Macro-F1 |
|---|---|---|
| Majority-class baseline | 0.648 | 0.197 |
| Direct 4-way (2a) | 0.671 | **0.535** |
| Two-stage cascade (2c) | 0.690 | 0.515 |
| FLAN-T5 zero-shot (2b) | 0.634 | 0.448 |
| FLAN-T5 few-shot, k=1 (2b) | 0.624 | 0.414 |

Findings worth noting:

- **Fine-tuning beats prompting by a wide margin** on this task. FLAN-T5
  zero-shot clears the majority baseline but trails a fine-tuned BERTweet by
  ~0.09 macro-F1.
- **More demonstrations made few-shot worse, not better.** Dev macro-F1 on the
  tuning subset fell from 0.43 at k=1 to 0.07 at k≥2 — longer prompts collapsed
  the model onto a single label rather than improving discrimination.
- **The cascade did not beat direct classification.** Routing only ~27% of dev
  examples to the second stage, its stage-wise gains did not survive
  composition — errors in comment detection are unrecoverable downstream.
- **`Deny` is the bottleneck throughout**, with the fewest training examples and
  the lowest per-class F1 in every approach.

### A note on the test split

Test-set numbers are **not** reported above, and shouldn't be read from the
notebook outputs as-is. Every approach returns an identical test macro-F1 of
0.213 — exactly the majority-class baseline — and the cascade's stage-1 output
is constant across all 1049 test rows (min = mean = max = 0.379). This is a
defect in the processed test CSV, not in the models: the text columns are not
reaching the tokeniser, so every model degenerates to predicting `Comment`.
Dev results are unaffected. Diagnosing this is outstanding work.

## Data

The experiments use the official RumourEval 2017 Subtask A train/dev/test
splits. **The dataset is not included in this repository** and must be obtained
from the organisers: https://alt.qcri.org/semeval2017/task8/

The notebooks read pre-processed CSVs from `data/processed/`
(`train.csv`, `dev.csv`, `test.csv`), each with `source_text`, `reply_text` and
`label` columns, flattening the original threaded JSON into source–reply pairs.

## Running

```bash
pip install -r requirements.txt
jupyter lab
```

Notebooks are committed with outputs. `1.ipynb` and `2b.ipynb` run on CPU;
`2a.ipynb` and `2c.ipynb` expect a CUDA device for practical training times.
Seeds are fixed, though exact reproduction across different hardware is not
guaranteed.

## Context

University coursework (COMP4167 Natural Language Processing). `Report.pdf` is
the accompanying write-up, in which the notebooks are referred to by their
question numbers. Experiments were run under constrained compute, so model
scale and hyperparameter search are deliberately limited. ChatGPT was used for
code debugging during development.
