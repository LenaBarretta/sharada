# Training the base models

Three files:

| | |
| --- | --- |
| [`sources.py`](sources.py) | the label sets, where their text and labels are, and how the question is asked |
| [`run.py`](run.py) | the run: the mix, the step loop, checkpoints, the measurement |
| [`publish.py`](publish.py) | the model card, written out of `report.json`, and the upload |
| [`kaggle.ipynb`](kaggle.ipynb) | the same run as a notebook, for a free T4 |

```bash
python training/run.py --encoder answerdotai/ModernBERT-base  --out runs/base
python training/run.py --encoder answerdotai/ModernBERT-large --out runs/large --batch-size 8 --lr 2e-5

HF_TOKEN=... python training/publish.py runs/base --repo LenaBarretta/sharada-base
```

A run writes `model.safetensors`, `config.json`, the tokenizer, `passport.json` (what the calibration
was fitted on), `report.json` (every number, per label set, plus the whole mix as declared) and
`checkpoint.pt`. Run it again against the same `--out` and it carries on from the checkpoint — the step
counter, the optimiser and the schedule included; `--fresh` ignores it.

## Trying it before spending an hour on it

```bash
python training/run.py --only news-section spam grammatical --steps 20 --batch-size 4 \
    --cap-scale 0.01 --eval-examples 40 --out /tmp/try
```

That is the whole pipeline — data, loop, checkpoint, calibration, report — in a couple of minutes on a
laptop, which is the cheap way to find out that something is wrong with it.

## Adding a label set

A `Source` is a declaration; add one to `SOURCES` and it joins the mix:

```python
Source("my-task", "me/my-dataset", "text", "label",
       ("How should this be handled?",            # several phrasings, picked at random per example
        "What should happen to this one?"),
       kind="choice", options=("ignore", "answer", "escalate"), cap=2000)
```

* `options` spells the labels out in plain words, in the order of the dataset's numeric label column.
  Leave it out and the names come from the dataset itself (a `ClassLabel`, or a `*_text` column),
  prettified; `rename` fixes the ones that arrive as codes.
* `asks_about` names a second column that goes **into the question** — that is how entailment and
  paraphrase tasks are posed here, and it is what teaches the model to read the question.
* `kind="scale"` means the options are ordered: they are then never shuffled and never sampled.
* `holdout=True` keeps the label set out of training. It is only ever measured, which is where the
  zero-shot numbers in the model cards come from.
* A source that fails to load is skipped with a line saying why. Dataset ids on the Hub move, and a
  three-hour run should not die for one of them — check the skipped list at the top of the log.

## Running it for quality rather than for the clock

A Kaggle session is cut off at 12 hours, so a long run is given a little less than that and finishes
itself: `--max-hours 11` stops the loop, keeps the best weights, fits the temperatures, writes the
report and leaves a checkpoint behind.

```bash
python training/run.py --encoder answerdotai/ModernBERT-base --out runs/base \
    --steps 40000 --batch-size 16 --accumulate 2 --cap-scale 3 --variants 2 \
    --eval-every 2000 --eval-examples 500 --max-hours 11
```

What each of those is for:

* `--steps` **is the plan, not the clock.** The cosine schedule is laid out over it; a run cut off at
  60% of its schedule ends at a high learning rate and is worse than a shorter schedule that finished.
  The log prints a measured rate and an eta within the first hundred steps — if it does not fit, stop
  and start again with a smaller `--steps` and `--fresh`.
* `--accumulate` buys an effective batch without the memory of one. 16 × 2 on a T4 is the same gradient
  as a batch of 32 and fits where 32 does not.
* `--cap-scale` and `--variants` are two different kinds of more data: `--cap-scale` reaches further
  into the big datasets, `--variants` draws each label set again with a different wording of its
  question and a different subset of its options. The small label sets only grow through the second
  one, and the wording is what the model is meant to be reading.
* `--keep best` (the default) finishes on the weights with the lowest held-out log loss rather than the
  last step's. On a mixture the two are rarely the same: one label set is still improving while another
  has started to overfit.
