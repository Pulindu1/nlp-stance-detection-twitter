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

## Known defect in the test split

The `test.csv` behind the committed notebook outputs was exported **with empty
text fields**. Labels and IDs survived; the text did not:

| Split | Rows | Columns | `source_text` | `reply_text` |
|---|---|---|---|---|
| `train.csv` | 3569 | 22 | populated | populated |
| `dev.csv` | 950 | 22 | populated | populated |
| `test.csv` | 1049 | 4 | **empty in all 1049 rows** | **empty in all 1049 rows** |

Every test-set number in the notebooks and in `report.pdf` therefore reflects
classifying an empty string, not model generalisation. Dev results are
unaffected. See "The test split is unusable" in the top-level README.

Because `tweet_id` and `label` survived, the split can be repaired by
re-deriving the text from the original SemEval distribution and joining on
`tweet_id`, without needing to redo the annotation or the splits.

## Validating before you train

The defect above would have been caught before any model ran by asserting on
the inputs at load time. Any regeneration of these CSVs should pass:

```python
import pandas as pd

REQUIRED = ["tweet_id", "source_text", "reply_text", "label"]

for split in ["train", "dev", "test"]:
    df = pd.read_csv(f"data/processed/{split}.csv")

    missing = [c for c in REQUIRED if c not in df.columns]
    assert not missing, f"{split}: missing columns {missing}"

    for col in ["source_text", "reply_text"]:
        blank = df[col].isna() | (df[col].astype(str).str.strip() == "")
        assert not blank.any(), f"{split}: {blank.sum()}/{len(df)} rows blank in {col}"

    assert set(df["label"].unique()) <= {"S", "D", "Q", "C"}, f"{split}: unexpected labels"
    print(f"{split}: {len(df)} rows OK")
```

All three splits should also expose the *same* columns. In the original CSVs
they did not, and `03_prompting_flan_t5.ipynb` — which resolves column names by
preference order — silently fed normalised text on train/dev but raw text on
test:

```
[train] src='source_text_norm' rpl='reply_text_norm' lbl='label'
[dev]   src='source_text_norm' rpl='reply_text_norm' lbl='label'
[test]  src='source_text'      rpl='reply_text'      lbl='label'
```

This is secondary to the empty-text defect, but is a second reason to make the
splits schema-identical when regenerating them.
