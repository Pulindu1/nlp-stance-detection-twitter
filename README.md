# Stance Detection on Twitter Rumour Threads

Four approaches to SDQC stance detection on **RumourEval 2017**
(SemEval-2017 Task 8, Subtask A), compared under severe class imbalance:
lexical and topic analysis, a fine-tuned transformer, an instruction-prompted
generative model, and a two-stage chain-of-thought cascade.

Given a source tweet making a claim and a reply to it, classify the reply's
stance toward the claim:

| Label | Stance | Example behaviour |
|---|---|---|
| **S** | Support | Agrees with or corroborates the rumour |
| **D** | Deny | Disagrees with or refutes it |
| **Q** | Query | Asks for clarification or evidence |
| **C** | Comment | Responds without a clear stance |

The central difficulty is imbalance: `Comment` is 64% of training replies
(C=2291, S=743, D=273, Q=262), so accuracy is close to meaningless and
**macro-F1** is the metric throughout. The splits are small too — 3569 train,
950 dev, 1049 test.

## Repository contents

```
notebooks/     the four experiments, committed with outputs
data/          dataset acquisition instructions and expected schema
               (the data itself is not redistributed)
```

| File | Contents |
|---|---|
| `notebooks/01_data_analysis.ipynb` | Top unigrams/bigrams per stance class, source-vs-reply token distribution contrasts, and LDA topic models with word clouds fitted separately to stance-bearing and Comment replies |
| `notebooks/02_direct_4way_bertweet.ipynb` | Direct 4-way fine-tuning of BERTweet with layer-wise LR decay, square-root inverse-frequency class weighting, label smoothing, and early stopping on dev macro-F1 |
| `notebooks/03_prompting_flan_t5.ipynb` | FLAN-T5-large zero-shot and few-shot prompting, scored by ranked log-probability over label tokens `A/B/C/D`, plus a per-class logit calibration sweep |
| `notebooks/04_two_stage_cascade.ipynb` | Two-stage cascade: binary Comment-vs-Non-Comment detection, then S/D/Q classification on routed examples, with the routing threshold τ tuned on dev |
| `data/README.md` | How to obtain the dataset and the expected CSV schema |

Notebooks are committed with their outputs, so every figure and number below is
inspectable without rerunning anything.

## Results

All results below are on the **development set**. The test split shipped with
empty text fields and is unusable — see
[the test split](#the-test-split-is-unusable) for the diagnosis.

Macro-F1 is the headline metric. A majority-class predictor is included because
it scores 0.65 accuracy while being useless, which is the trap this task sets.

| Approach | Dev accuracy | Dev macro-F1 |
|---|---|---|
| Majority class (all `Comment`) | 0.65 | 0.20 |
| **Direct 4-way BERTweet** | 0.67 | **0.54** |
| Two-stage cascade | 0.69 | 0.52 |
| FLAN-T5 zero-shot | 0.63 | 0.45 |
| FLAN-T5 few-shot (k=1) | 0.62 | 0.41 |

Stage-wise, the cascade's components were individually strong — 0.662 macro-F1
for Comment detection, 0.636 for three-way S/D/Q — but those gains did not
survive composition.

Per-class dev F1 shows where the macro average comes from:

| Approach | S | D | Q | C |
|---|---|---|---|---|
| Direct 4-way BERTweet | 0.53 | 0.25 | 0.59 | 0.77 |
| FLAN-T5 zero-shot | 0.53 | 0.09 | 0.46 | 0.75 |

### What the comparison shows

**Fine-tuning beat prompting by a wide margin.** FLAN-T5 zero-shot clears the
majority baseline on dev but trails fine-tuned BERTweet by ~0.09 macro-F1.
Prompt-only classification with a general instruction-tuned model was not
competitive under this degree of imbalance.

**More demonstrations made few-shot worse.** Dev macro-F1 fell from 0.40 at
k=1 to ≈0.07 at k≥2 — while still inside the token limit, so this is not
truncation. Additional demonstrations appear to introduce conflicting patterns
rather than sharpening the decision boundary.

**Decomposition did not beat the direct classifier.** The cascade is
theoretically better suited to imbalance, since Stage 2 never sees the majority
class and can spend capacity on the subtle S/D/Q distinctions. In practice
hard routing at Stage 1 is unrecoverable: an example misrouted as `Comment`
can never be corrected downstream, and that error propagation outweighed the
benefit. A simpler end-to-end model won.

**`Deny` is the bottleneck everywhere.** With 273 training examples and the
most implicit linguistic realisation — sarcasm, indirect contradiction — it has
the lowest per-class F1 in every approach (as low as 0.09 for zero-shot
prompting). The n-gram analysis shows why keyword-style cues struggle: explicit
negation markers like "not" characterise only the most overt denials.

**Calibration did not transfer usefully.** Tuning per-class logit offsets on
dev improved few-shot only marginally (0.414 → 0.417), so the residual error is
not a simple recoverable class-prior shift.

### What the lexical analysis shows

The four classes separate on surface cues, but only partially. `Support`
replies echo factual detail from the source (place names, event hashtags);
`Deny` leans on explicit negation (`not`, "that's not", "not isis"); `Query` is
dominated by interrogatives (`what`, `why`, "why did"); `Comment` is generic
and conversational (`just`, `like`) plus event nouns. Contrasting token
probabilities between stance-bearing and Comment replies shows stance replies
skew towards event-specific entities and Comment replies towards social
reaction (`people`, `hope`).

LDA over the two groups was much weaker: eight topics fitted separately to
stance and Comment replies gave near-identical mean topic proportions, so
thematic content does not distinguish stance. Both analyses are bag-of-words
and ignore discourse structure, so they are suggestive rather than conclusive —
part of the case for using learned contextual representations instead.

## The test split is unusable

The notebooks contain test-set numbers (macro-F1 0.23 direct, 0.21 prompting,
0.21 cascade) and read them as a generalisation failure under distribution
shift. **That reading is wrong, and the test numbers should be disregarded.**
The split was exported with its text fields empty:

```
test.csv  — 1049 rows, 4 columns
  label       : 4 distinct values, 0 empty     ← intact
  source_text : 1 distinct value, 1049 empty   ← every row is ""
  reply_text  : 1 distinct value, 1049 empty   ← every row is ""
```

For comparison, `train.csv` and `dev.csv` have 22 columns with fully populated
text. Every model was therefore asked to classify an empty string 1049 times,
and the resulting scores describe the export bug rather than any model.

The tell is in `notebooks/04_two_stage_cascade.ipynb`, whose Stage 1 logs
`p(noncomment)` as `min = mean = max = 0.3785` across all 1049 rows. A genuinely
overfit classifier still produces *varying* probabilities on varying text;
identical probabilities to sixteen significant figures can only mean identical
input. The cascade consequently routed 0% of test examples, never invoked
Stage 2, and scored exactly the all-`Comment` baseline — which the notebook
attributes to Stage 1 routing errors.

Labels and `tweet_id`s survived the export, so the fix is to re-derive the text
from the original SemEval distribution and re-run. Until then, dev is the only
sound basis for comparison, which is why the results table above is dev-only.

Two lessons worth stating plainly, since this is the most instructive part of
the project: a metric that lands *exactly* on the majority-class baseline
deserves suspicion rather than a narrative explaining it, and input data should
be asserted on — a two-line non-empty check at load time would have caught this
before any model ran. `data/README.md` includes that check.

## Method notes

- **Preprocessing** follows BERTweet's pre-training conventions: URLs → `HTTPURL`,
  mentions → `@USER`, whitespace collapsed. Deliberately no stopword removal or
  aggressive filtering, since negation markers and interrogatives (`not`, `why`,
  `?`) are exactly the stance signal.
- **Imbalance mitigation**: square-root inverse-frequency class weights in the
  loss, chosen over full inverse frequency to avoid instability from the
  extreme weight the `Deny` class would otherwise receive.
- **Input format**: source and reply are fused as a sentence pair rather than
  concatenated, so the model can attend across the boundary.
- **Training**: AdamW with weight decay, linear schedule with warm-up, gradient
  clipping, raised dropout and label smoothing against over-confidence, early
  stopping on dev macro-F1 (best checkpoint at epoch 5). Loss falls steadily
  and flattens after roughly 1000 steps.
- **Prompting** uses ranked scoring over label tokens instead of free-form
  generation, which makes outputs deterministic and eliminates invalid or
  hallucinated labels by construction. Tokenisation uses left truncation and
  long texts are clipped to a fixed character budget inside the prompt, which
  costs some macro-F1 but keeps the source–reply pair intact. Few-shot
  demonstrations are hand-picked for clear cues (question marks for `Query`,
  explicit negation for `Deny`, agreement terms for `Support`).
- **Cascade**: Stage 1 is a binary BERTweet classifier over `Comment` vs
  `Non-Comment` trained on the full training set; an example is routed to
  Stage 2 unless `p(Comment | x) ≥ τ`. Stage 2 is a separate BERTweet trained
  only on S/D/Q, so it never spends capacity on the majority class. Routing is
  hard, so Stage 1 errors are terminal.
- **Reproducibility**: seeds fixed (`SEED = 42`); exact package versions in
  `requirements.txt`.

## Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Obtain the dataset first — see `data/README.md`, which also includes a
validation snippet worth running before you train. The notebooks resolve
`data/processed/` and their `runs/` output directory against the repository
root, so they work whether Jupyter is launched from the root or from
`notebooks/`.

Pinned versions in `requirements.txt` are those used to produce the committed
outputs (Python 3.14).

`01` and `03` run on CPU. `02` and `04` fine-tune BERTweet and want a CUDA
device; they fall back to CPU but slowly. Experiments were run under a
constrained compute budget, so model scale and hyperparameter search are
limited by design rather than by conclusion.

## Ethical considerations

**Consent and privacy.** Tweets are public but their authors did not consent to
research reuse, and replies can carry names, locations, or speculative
accusations. Tweets are treated purely as text here: no user-level metadata, no
identity inference, and outputs restricted to closed-set stance labels. Any
real deployment would need anonymisation and strict data minimisation on top.

**Stance is not veracity.** A reply's stance toward a claim says nothing about
whether the claim is true. Treating these predictions as credibility signals
would risk amplifying misinformation in exactly the high-stakes settings where
rumour analysis is tempting; fact-checking and human oversight remain separate
requirements.

**Bias and misuse.** `Comment` dominance biases every model towards the
majority class, under-representing denial and questioning — which is why
macro-F1 and per-class metrics are reported rather than accuracy. At scale,
stance classification is open to misuse for viewpoint monitoring or
suppression, particularly if misclassification falls unevenly on minority
groups, so the limitations above are documented rather than smoothed over.

## Acknowledgements

- Derczynski et al., *SemEval-2017 Task 8: RumourEval* —
  https://aclanthology.org/S17-2006.pdf
- Nguyen et al., *BERTweet: A pre-trained language model for English Tweets*
- Chung et al., *Scaling Instruction-Finetuned Language Models* (FLAN-T5)

## Licence

MIT — see `LICENSE`. The licence covers the code and analysis, not the
RumourEval dataset, which remains under the task organisers' terms.
