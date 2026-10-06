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

HF_TOKEN=... python training/publish.py runs/base --repo lenabarretta/sharada-base
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

## The multilingual run

`--multilingual` adds the label sets that exist in many languages and leaves the English ones in
place, so the model gains languages instead of trading English for them. Without the flag they are
skipped entirely, which is what keeps the English checkpoints reproducible.

```bash
python training/run.py --encoder jhu-clsp/mmBERT-base --out runs/multilingual --multilingual \
    --steps 15000 --batch-size 8 --accumulate 4 --cap-scale 3 --variants 2 --max-hours 11
```

mmBERT is ModernBERT with a 256k vocabulary: the same twenty-two layers and the same hidden size as
`ModernBERT-base`, so it costs about what `base` costs to compute, and the extra 157M parameters are
the embedding table — memory rather than arithmetic. There is no mmBERT-large, so there is no
multilingual `large` to train.

A source marked `multilingual=True` pools several language editions of one dataset under a single
task through `configs`; the cap is split between them, so adding a language widens the task instead
of enlarging it. That works only because the label names are identical in every edition — MASSIVE's
sixty intents, XNLI's three answers, SIB-200's seven topics — which is also what makes
`topic-unseen-languages` meaningful: it is SIB-200 in eight languages kept out of training, so the
label set is familiar and the language is not.

One honest simplification: a pooled task gets one temperature across all its languages, though
calibration genuinely differs between them. Per-language calibration is a refinement, not something
this run does.

## Half-precision weights

`save` writes float16 and `from_pretrained` casts back to float32, so a checkpoint takes half the
download and the model you load is the full-precision one. This is not quantisation: no layer is
replaced, no arithmetic changes, and fine-tuning works exactly as it did. Measured on `sharada-base`,
the round trip moves probabilities by 2.6e-4 and the rounding is idempotent — the second save changes
nothing, because the weights are already on the float16 grid.

The one case where it would bite: a fine-tune whose updates are smaller than the spacing of that grid
(around 1e-5 for a weight of 0.01) would be rounded away on save. Nothing at the learning rates here
comes close.

Real quantisation — int8 — was measured and rejected: on a laptop CPU it gave no speedup at all,
because at batch 1 the time goes on launching kernels rather than on arithmetic, and it moved
probabilities by 0.069, which for a model sold on its calibration is not a rounding error. If it is
ever wanted for throughput, it belongs in a separate, inference-only artifact with its temperatures
fitted again on the quantised model and its calibration error measured afresh.
