<p align="center">
  <img src="https://raw.githubusercontent.com/LenaBarretta/sharada/main/docs/logo.png" alt="Sharada" width="200">
</p>

<h1 align="center">Sharada</h1>

<p align="center">
  <em>Typed decisions about text in one forward pass — options in the request, calibrated probabilities out.</em>
</p>

<p align="center">
  <a href="https://github.com/LenaBarretta/sharada/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="Apache 2.0"></a>
  <a href="https://huggingface.co/lenabarretta/sharada-base"><img src="https://img.shields.io/badge/%F0%9F%A4%97-sharada--base-yellow" alt="sharada-base on the Hub"></a>
  <a href="https://lenatriestounderstand.com/notes/llm/024-rlcr/"><img src="https://img.shields.io/badge/write--up-lenatriestounderstand-ff69b4" alt="the write-up"></a>
</p>

A small encoder that makes **typed decisions about text in one forward pass**: the options come with the
request, and what comes back is a probability for each of them. Nothing is generated, nothing is parsed,
and the same model answers a question it has never seen because the labels are part of the input rather
than part of the weights.

```python
from sharada import DecisionModel

model = DecisionModel.from_pretrained("lenabarretta/sharada-base")

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

| model | parameters | accuracy | ECE | one decision | |
| --- | --- | --- | --- | --- | --- |
| `lenabarretta/sharada-base` | 150M | 0.813 | 0.008 | 19.8 ms | the default; fine-tune this one |
| `lenabarretta/sharada-large` | 397M | 0.834 | 0.009 | 26.6 ms | better, and better still on label sets it has never seen |
| `lenabarretta/sharada-multilingual-base` | 308M | 0.802 | 0.009 | 23.1 ms | the same model over many more languages, on mmBERT |
| `lenabarretta/sharada-multilingual-small` *(in training)* | 140M | | | | a narrower body: for throughput, not for one fast answer |

Weights are stored in half precision: the encoder was trained under a float16 autocast, so the bits
below that were never signal, and the file halves for a shift in the probabilities of about 2e-4.
Loading casts back to float32, so fine-tuning is unaffected — this is a storage format, not
quantisation, and nothing about the model you get is approximate.

Accuracy is over 38 label sets with every label offered at once — banking on all 77 intents, clinc on
all 151 — and **ECE** is how far the stated probability lands from how often it turns out right. The
gap between the two models is widest where it matters most: on label sets neither was trained on,
`large` reads arXiv categories at 0.456 against `base`'s 0.317.

A checkpoint carries its own encoder, limits and temperatures in `config.json`, so a bigger model — or a
multilingual one, built on a multilingual encoder — is another repository rather than another version of
the library. It also means a multilingual model cannot be a flag on an English one: this encoder is
English down to its tokenizer, and another language means other weights.

Accuracy and calibration per label set are in each model card, measured on label sets the model was
**not** trained on as well as on the ones it was.

## What it will not do

- **It does not generate.** No free-form answers, no extraction, no reasoning out loud. Options or
  nothing.
- **256 tokens of text** by default (`Limits`), with 48 for the question and 12 per option. Longer
  documents need chunking or a larger limit, and the limit costs quadratic attention.
- **One of them is English-only.** `base` and `large` are built on an English encoder and were
  trained on English label sets; the multilingual checkpoints cover far more, but pay about two and a
  half points of English accuracy for it, and no checkpoint has been measured on a language outside
  the forty-odd in the mix.
- **No medicine, no law, no code.** Nothing of the sort is in the training mix. The two medical label
  sets are measured only, and one of them — verifying public-health claims — lands at the
  majority-class baseline, which is to say it does not work. Those domains need fine-tuning on your
  own labelled examples.
- **It is small.** Where a frontier model knows a fact that is not in the text, it wins. This answers
  questions about the text in front of it.

## What each one scores, label set by label set

Overall numbers hide the thing worth knowing: the models differ most where the task is hardest, and
barely at all where it is easy. Rows marked **unseen** were kept out of training entirely — those are
the zero-shot numbers. A dash means the label set is not in that model's mix.

<!-- per-label-set table, generated by training/compare.py -->
| label set | answer options | trained on | measured on | base | large | multilingual | what it is |
| --- | --- | --- | --- | --- | --- | --- | --- |
| clinc-intent | 151 | 24,000 | 480 | 0.908 | 0.950 | 0.883 | `clinc/clinc_oos/plus` |
| banking-intent | 77 | 19,986 | 480 | 0.879 | 0.898 | 0.854 | `mteb/banking77` |
| massive-intent | 59 | 23,028 | 480 | 0.867 | 0.879 | 0.875 | `mteb/amazon_massive_intent/en` |
| question-type-fine | 50 | 10,904 | 240 | 0.912 | 0.904 | 0.900 | `CogComp/trec` |
| massive-intent-multi | 35 | 72,000 | 1,440 | — | — | 0.849 | `mteb/amazon_massive_intent ×18 languages` |
| fine-emotion | 28 | 18,000 | 360 | 0.583 | 0.575 | 0.581 | `google-research-datasets/go_emotions/simplified` |
| newsgroup | 20 | 14,592 | 292 | 0.695 | 0.726 | 0.685 | `SetFit/20_newsgroups` |
| entity-type | 14 | 18,000 | 360 | 0.992 | 0.997 | 0.997 | `fancyzhx/dbpedia_14` |
| forum-topic | 10 | 18,000 | 360 | 0.756 | 0.789 | 0.769 | `community-datasets/yahoo_answers_topics` |
| topic-multi | 7 | 30,398 | 660 | — | — | 0.815 | `mteb/sib200 ×24 languages` |
| question-type | 6 | 10,904 | 240 | 0.979 | 0.963 | 0.975 | `CogComp/trec` |
| emotion | 6 | 15,000 | 300 | 0.897 | 0.910 | 0.913 | `dair-ai/emotion` |
| sentence-tone | 5 | 15,000 | 300 | 0.617 | 0.637 | 0.570 | `SetFit/sst5` |
| review-stars | 5 | 24,000 | 480 | 0.646 | 0.658 | 0.650 | `Yelp/yelp_review_full` |
| app-stars | 5 | 15,000 | 300 | 0.720 | 0.717 | 0.677 | `sealuzh/app_reviews` |
| tweet-emotion | 4 | 6,514 | 240 | 0.846 | 0.879 | 0.817 | `cardiffnlp/tweet_eval/emotion` |
| news-section | 4 | 18,000 | 360 | 0.942 | 0.931 | 0.933 | `fancyzhx/ag_news` |
| tweet-sentiment | 3 | 15,000 | 300 | 0.693 | 0.713 | 0.680 | `cardiffnlp/tweet_eval/sentiment` |
| entailment-short | 3 | 18,000 | 360 | 0.892 | 0.906 | 0.875 | `stanfordnlp/snli` |
| entailment-multi | 3 | 71,968 | 1,428 | — | — | 0.764 | `facebook/xnli ×14 languages` |
| entailment | 3 | 24,000 | 480 | 0.833 | 0.873 | 0.815 | `nyu-mll/glue/mnli` |
| toxic-comment | 2 | 15,000 | 300 | 0.930 | 0.933 | 0.920 | `SetFit/toxic_conversations` |
| spam | 2 | 10,034 | 240 | 0.992 | 0.992 | 0.992 | `ucirvine/sms_spam` |
| short-verdict | 2 | 12,000 | 240 | 0.929 | 0.950 | 0.938 | `cornell-movie-review-data/rotten_tomatoes` |
| same-question | 2 | 18,000 | 360 | 0.831 | 0.867 | 0.825 | `nyu-mll/glue/qqp` |
| same-meaning | 2 | 7,336 | 240 | 0.817 | 0.808 | 0.787 | `nyu-mll/glue/mrpc` |
| product-tone | 2 | 15,000 | 300 | 0.957 | 0.960 | 0.947 | `fancyzhx/amazon_polarity` |
| paraphrase | 2 | 15,000 | 300 | 0.907 | 0.950 | 0.883 | `google-research-datasets/paws/labeled_final` |
| offensive | 2 | 15,000 | 300 | 0.847 | 0.860 | 0.863 | `cardiffnlp/tweet_eval/offensive` |
| movie-verdict | 2 | 15,000 | 300 | 0.930 | 0.957 | 0.933 | `stanfordnlp/imdb` |
| irony | 2 | 5,724 | 240 | 0.733 | 0.787 | 0.725 | `cardiffnlp/tweet_eval/irony` |
| hateful | 2 | 14,989 | 300 | 0.833 | 0.837 | 0.810 | `cardiffnlp/tweet_eval/hate` |
| grammatical | 2 | 15,000 | 300 | 0.790 | 0.803 | 0.740 | `nyu-mll/glue/cola` |
| follows | 2 | 4,980 | 240 | 0.817 | 0.875 | 0.779 | `nyu-mll/glue/rte` |
| answers-question | 2 | 18,000 | 360 | 0.894 | 0.942 | 0.892 | `nyu-mll/glue/qnli` |
| massive-scenario *(unseen)* | 18 | — | 240 | 0.733 | 0.742 | 0.767 | `mteb/amazon_massive_scenario/en` |
| arxiv-category *(unseen)* | 11 | — | 180 | 0.317 | 0.456 | 0.522 | `ccdv/arxiv-classification/no_ref` |
| topic-unseen-languages *(unseen)* | 7 | — | 240 | — | — | 0.679 | `mteb/sib200 ×8 languages` |
| poem-tone *(unseen)* | 4 | — | 96 | 0.375 | 0.573 | 0.510 | `google-research-datasets/poem_sentiment` |
| claim-veracity *(unseen)* | 4 | — | 180 | 0.544 | 0.467 | 0.383 | `ImperialCollegeLondon/health_fact` |
| subjective *(unseen)* | 2 | — | 180 | 0.617 | 0.683 | 0.461 | `SetFit/subj` |
| medical-pair *(unseen)* | 2 | — | 180 | 0.739 | 0.772 | 0.750 | `curaihealth/medical_questions_pairs` |
| **overall** | | **663,357** | **15,256** | **0.813** | **0.834** | **0.802** | |
<!-- per-label-set table, generated by training/compare.py -->

Per-label-set calibration error is in each model card; here the question is only which model answers
more of them right.

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

model = DecisionModel.from_pretrained("lenabarretta/sharada-base")
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

## Train any of them yourself

[`training/`](https://github.com/LenaBarretta/sharada/tree/main/training) has the whole run. The mix of
label sets is declared in
[`sources.py`](https://github.com/LenaBarretta/sharada/blob/main/training/sources.py) — intents, topics,
review scores, emotion, toxicity, spam, entailment — each one asked through several wordings of its
question, with the options shuffled and long label sets often shown as a sampled handful, so the model
learns to read the options rather than their positions. Some label sets are kept out of training
entirely and only measured: those are the zero-shot numbers above.

Every published checkpoint comes from one of these four commands, with the settings they share
written out in full so that nothing about them has to be guessed:

```bash
SHARED="--steps 15000 --cap-scale 3 --variants 2 --max-hours 11 \
        --eval-every 1000 --eval-examples 300 --checkpoint-every 500 --keep best"

python training/run.py --encoder answerdotai/ModernBERT-base  --out runs/base  \
    --batch-size 16 --accumulate 2 --lr 3e-5 $SHARED
python training/run.py --encoder answerdotai/ModernBERT-large --out runs/large \
    --batch-size 8 --accumulate 4 --lr 2e-5 $SHARED
python training/run.py --encoder jhu-clsp/mmBERT-base  --out runs/multilingual-base  --multilingual \
    --batch-size 8 --accumulate 4 --lr 3e-5 $SHARED
python training/run.py --encoder jhu-clsp/mmBERT-small --out runs/multilingual-small --multilingual \
    --batch-size 16 --accumulate 2 --lr 3e-5 $SHARED
```

The effective batch is 32 in all four; the micro-batch differs only because a wider model and a larger
vocabulary need more memory to hold. `--multilingual` adds the label sets that come in many languages
and keeps the English ones, so a multilingual model gains languages rather than trading English for
them; without the flag they are skipped, which is what keeps the English checkpoints reproducible.

Measured on one free T4: about three hours for `base`, four to five for `large`, five for
`multilingual-base`. The multilingual runs also spend twenty minutes assembling the mix before the
first step, because it downloads sixty-four language editions.

A multilingual encoder costs more to train than its body suggests. mmBERT-base has exactly the same
twenty-two layers and hidden size as ModernBERT-base, so a forward pass is the same work — but the
optimiser updates every parameter on every step, and 197M of its 287M are the embedding table. That is
twice the optimiser traffic for the same arithmetic, which is where the extra two hours go. At
inference the table costs nothing: looking a row up is not a matrix multiply.

The run checkpoints every five hundred steps and resumes from a checkpoint if it finds one, which is
what lets a Kaggle session that ran out of time be restarted rather than lost;
[`training/kaggle.ipynb`](https://github.com/LenaBarretta/sharada/blob/main/training/kaggle.ipynb) is
the notebook wrapper, and a word in its first cell picks which of the four to train. What resuming
cannot do is extend a run that already finished: the learning-rate schedule is laid out over the total
number of steps, so asking for more lifts it back up, and
[`training/README.md`](https://github.com/LenaBarretta/sharada/blob/main/training/README.md) has what
happened when that was tried.

## Where it came from

The design, the experiments behind it and what each training signal did are written up in
[RLCR from Scratch](https://lenatriestounderstand.com/notes/llm/024-rlcr/), with a runnable lab.

Apache 2.0.
