# Data

The dataset is **not included in this repository**. RumourEval 2017 is
distributed by the task organisers under their own terms, so only the
preparation contract is documented here.

## Obtaining the data

1. Download RumourEval 2017 Subtask A from the organisers:
   - Task page: https://alt.qcri.org/semeval2017/task8/
   - Data and tools: https://alt.qcri.org/semeval2017/task8/index.php?id=data-and-tools
   - Task description paper: https://aclanthology.org/S17-2006.pdf
2. Use the official train / dev / test splits without modification.
3. Flatten the threaded JSON into one row per (source, reply) pair and write
   `train.csv`, `dev.csv`, `test.csv` into `data/processed/`.

## Expected schema

The notebooks read `data/processed/{train,dev,test}.csv`. The columns actually
required are:

| Column | Description |
|---|---|
| `tweet_id` | Identifier of the reply tweet |
| `source_text` | Text of the source tweet making the claim |
| `reply_text` | Text of the reply whose stance is being classified |
| `label` | Gold stance: one of `S`, `D`, `Q`, `C` |

Split sizes used in these experiments: train 3569, dev 950, test 1049.

## Known inconsistency in the splits used

The CSVs behind the committed notebook outputs were **not uniform across
splits**, and this is load-bearing for how the test results should be read:

- `train.csv` and `dev.csv` had 22 columns, including pre-normalised
  `source_text_norm` / `reply_text_norm` variants and a range of derived
  features.
- `test.csv` had only 4 columns, with no `*_norm` variants.

`03_prompting_flan_t5.ipynb` resolves its input columns by preference order and
logs what it picked:

```
[train] src='source_text_norm' rpl='reply_text_norm' lbl='label'
[dev]   src='source_text_norm' rpl='reply_text_norm' lbl='label'
[test]  src='source_text'      rpl='reply_text'      lbl='label'
```

So that notebook fed **normalised** text on train/dev and **raw** text on test.
Any regeneration of these CSVs should produce identical columns for all three
splits, which removes this asymmetry. See the "Reading the test results"
section of the top-level README for why it matters.
