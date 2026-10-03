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
