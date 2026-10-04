<p align="center">
  <img src="https://raw.githubusercontent.com/LenaBarretta/sharada/main/docs/logo.png" alt="Sharada" width="200">
</p>

<h1 align="center">Sharada</h1>

<p align="center">
  <em>Typed decisions about text in one forward pass — options in the request, calibrated probabilities out.</em>
</p>

<p align="center">
  <a href="https://github.com/LenaBarretta/sharada/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0"></a>
  <a href="https://huggingface.co/LenaBarretta/sharada-base"><img src="https://img.shields.io/badge/%F0%9F%A4%97-sharada--base-yellow" alt="sharada-base on the Hub"></a>
  <a href="https://lenatriestounderstand.com/notes/llm/024-rlcr/"><img src="https://img.shields.io/badge/write--up-lenatriestounderstand-ff69b4" alt="the write-up"></a>
</p>

A small encoder that makes **typed decisions about text in one forward pass**: the options come with the
request, and what comes back is a probability for each of them. Nothing is generated, nothing is parsed,
and the same model answers a question it has never seen because the labels are part of the input rather
than part of the weights.

```python
from sharada import DecisionModel

model = DecisionModel.from_pretrained("LenaBarretta/sharada-base")

d = model.decide(
    text="My card still hasn't arrived and I ordered it two weeks ago.",
    question="Which team should handle this?",
    options=["billing", "card delivery", "technical support", "account closure"],
)

d.answer          # 'card delivery'
d.confidence      # 0.86
d.probabilities   # {'billing': 0.07, 'card delivery': 0.86, ...}
```

Three kinds of question are typed, so the model knows what the options mean:

| kind | the options are | example |
| --- | --- | --- |
| `choice` | unordered labels | which team, which topic, which intent |
| `scale` | ordered steps | severity 1–5, how positive, how urgent |
| `binary` | yes or no | is this spam, does this need a human |

```python
from sharada import Request

model.decide_many([
    Request(text=review, question="How positive is this review?",
            options=["very negative", "negative", "neutral", "positive", "very positive"],
            kind="scale", task="review-tone"),
    Request(text=review, question="Does this mention a refund?",
            options=["no", "yes"], kind="binary", task="refund-flag"),
])
```

## Install

```bash
pip install git+https://github.com/LenaBarretta/sharada
```

Python 3.10+, `torch` and `transformers`; CPU is enough to run it.

## Models

| model | encoder | parameters | what it is for |
| --- | --- | --- | --- |
| `LenaBarretta/sharada-base` | ModernBERT-base | 150M | the default; fine-tune this one |
| `LenaBarretta/sharada-large` *(in training)* | ModernBERT-large | 400M | a few points better, ~2.5× the time |
| `LenaBarretta/sharada-multilingual` *(next)* | mmBERT | 300M | the same architecture over 1800+ languages |

A checkpoint carries its own encoder, limits and temperatures in `config.json`, so a bigger model — or a
multilingual one, built on a multilingual encoder — is another repository rather than another version of
the library. It also means a multilingual model cannot be a flag on an English one: this encoder is
English down to its tokenizer, and another language means other weights.

Accuracy and calibration per label set are in each model card, measured on label sets the model was
**not** trained on as well as on the ones it was.

## Three things the architecture guarantees

The request is laid out as one sequence — text, question, then every option as a parallel branch — and a
mask decides who may read whom. Both are in [`layout.py`](https://github.com/LenaBarretta/sharada/blob/main/sharada/layout.py) and
[`masking.py`](https://github.com/LenaBarretta/sharada/blob/main/sharada/masking.py), and they buy three properties that hold by construction, not because
training got them approximately right:

1. **The order of the options cannot matter.** Every option branch starts at the same position id, and
   the encoder's positions are rotary, so no option is earlier or later than another. Reshuffle them and
   the probabilities follow their options exactly.
2. **An option's score does not depend on which other options are offered.** An option reads the text,
   the question and itself, nothing else. Drop two options from a list of four and the remaining two keep
   their scores to the last bit — so the probabilities are a renormalisation, and a long list of options
   does not make each one noisier.
3. **The text is read once.** Text tokens read only text tokens, so their states do not depend on the
   question. Ten questions about one document are ten cheap read-outs over one encoding of it.

These are the tests in [`tests/test_model.py`](https://github.com/LenaBarretta/sharada/blob/main/tests/test_model.py), checked on an untrained model.

## Fine-tune it on your own labels

This is what the package is built around: a few hundred labelled examples and a few minutes.

```python
from sharada import DecisionModel, Example

examples = [
    Example(text="the invoice is wrong again", question="Which team should handle this?",
            options=["billing", "technical", "sales"], label=0, task="routing"),
    ...
]

model = DecisionModel.from_pretrained("LenaBarretta/sharada-base")
report = model.fit(examples)        # holds out 20%, stops when held-out log loss stops improving
model.save("my-router")

report        # Report(820 examples, accuracy 0.914, log loss 0.287, calibration error 0.031)
```

`fit` keeps a part of the examples out, trains on the rest, early-stops, then fits one temperature per
task on the held-out part and writes a calibration passport. Useful arguments:

| argument | default | |
| --- | --- | --- |
| `loss` | `"cross_entropy"` | `"brier"` scores the whole distribution, not just the right option |
| `freeze_encoder` | `False` | train the read-out only: seconds, and enough for a few hundred examples |
| `batch_size`, `lr`, `max_epochs`, `patience` | `16`, `2e-5`, `10`, `2` | |

Mixing several tasks in one `fit` is the normal case — give each one its own `task` name and each gets
its own temperature. See [`examples/finetune_your_own.py`](https://github.com/LenaBarretta/sharada/blob/main/examples/finetune_your_own.py), which trains
on a CSV.

## The number next to the answer is supposed to be true

`0.86` should mean right about 86% of the time. That is a property of a distribution, not of a model, so
it is fitted and it expires:

```python
from sharada import check_passport, save_passport

passport = model.calibrate(recent_examples)     # one temperature per task
save_passport(passport, "passport.json")

check_passport(passport, recent_examples, model)
# ['the calibration expired on 2026-04-01; fit it again on recent answers',
#  'routing: the options have changed since the calibration']
```

`check_passport` returns an empty list when it finds nothing wrong — run it in the deployment pipeline
and fail the build on anything it returns. `evaluate` gives accuracy, log loss, Brier, expected
calibration error with a bootstrap interval, a reliability curve and a risk–coverage curve.

## What to do with 0.86

A probability is not a decision. `Policy` turns one into the action that costs the least, including
handing the request to a person:

```python
from sharada import Policy, escalation_budget

policy = Policy(options=["approve", "reject"],
                costs={("fraud", "approve"): 10_000, ("clean", "reject"): 100},
                escalate=30)

policy.act({"fraud": 0.1, "clean": 0.9})     # Action('answer', 'reject', expected_loss=90.0)
policy.act({"fraud": 0.5, "clean": 0.5})     # Action('escalate', ...)

# a person can look at 5% of the traffic, no more:
policy = escalation_budget(policy, probabilities, budget=0.05)
```

## Train the base models yourself

[`training/`](https://github.com/LenaBarretta/sharada/tree/main/training) has the whole run: a mix of public label sets in
[`sources.py`](https://github.com/LenaBarretta/sharada/blob/main/training/sources.py) — intents, topics, sentiment, toxicity, spam, entailment, review
scores — each one presented with several question phrasings, shuffled options and sampled option
subsets, so the model learns to read the options rather than their positions. Some label sets are held
out of training entirely and only measured, which is where the zero-shot numbers come from.

```bash
python training/run.py --encoder answerdotai/ModernBERT-base --out runs/base
```

It checkpoints every few hundred steps and resumes from the checkpoint if it finds one, which is what
makes it survive a Kaggle session; [`training/kaggle.ipynb`](https://github.com/LenaBarretta/sharada/blob/main/training/kaggle.ipynb) is the notebook
wrapper. One free T4: about an hour for base, about three for large.

## What it will not do

- **It does not generate.** No free-form answers, no extraction, no reasoning out loud. Options or
  nothing.
- **256 tokens of text** by default (`Limits`), with 48 for the question and 12 per option. Longer
  documents need chunking or a larger limit, and the limit costs quadratic attention.
- **English.** The encoder is English-only; a multilingual encoder drops in, but the published
  checkpoints are not trained for it.
- **It is small.** Where a frontier model knows a fact that is not in the text, it wins. This answers
  questions about the text in front of it.

## Where it came from

The design, the experiments behind it and what each training signal did are written up in
[RLCR from Scratch](https://lenatriestounderstand.com/notes/llm/024-rlcr/), with a runnable lab.

Apache 2.0.
