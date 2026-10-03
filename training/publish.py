"""Push a finished run to the Hub, with a model card made out of what was measured.

    HF_TOKEN=... python training/publish.py runs/base --repo LenaBarretta/sharada-base

The card is written from `report.json`, so the numbers on the Hub are the numbers the run produced and
nobody has to keep them in step by hand. Everything in the run directory goes up except the training
checkpoint, which is large and of no use to anyone downloading the model.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

CARD = """---
license: apache-2.0
library_name: sharada
pipeline_tag: zero-shot-classification
tags:
- text-classification
- zero-shot-classification
- calibration
- uncertainty
- decision-making
- modernbert
base_model: {encoder}
---

# {repo}

A small encoder that makes **typed decisions about text in one forward pass**. The options come with the
request and what comes back is a probability for each of them: nothing is generated and nothing is
parsed. Because the labels are part of the input rather than part of the weights, a label set it has
never been trained on still gets an answer.

Code, examples and the training run: <https://github.com/LenaBarretta/sharada>.
The design and the experiments behind it: <https://lenatriestounderstand.com/notes/llm/024-rlcr/>.

```bash
pip install sharada
```

```python
from sharada import DecisionModel

model = DecisionModel.from_pretrained("{repo}")

d = model.decide(
    text="My card still hasn't arrived and I ordered it two weeks ago.",
    question="Which team should handle this?",
    options=["billing", "card delivery", "technical support", "account closure"],
)
d.answer, d.confidence, d.probabilities
```

Fine-tuning on a few hundred of your own labelled examples is the intended use:

```python
report = model.fit(examples)      # holds 20% out, early-stops, then fits a temperature per task
model.save("my-router")
```

## What it is

| | |
| --- | --- |
| encoder | `{encoder}` |
| parameters | {parameters} |
| text | up to {text_tokens} tokens, plus 48 for the question and 12 per option |
| kinds of question | `choice` (unordered labels), `scale` (ordered steps), `binary` (yes or no) |
| output | one probability per option, from a single forward pass |
| one decision | {ms_per_question} ms per question{device_note} |

Three properties hold by construction rather than by training: the **order of the options cannot change
the answer** (every option branch starts at the same position id), an **option's score does not depend
on which other options are offered** (an option reads the text, the question and itself), and the
**text is read once** however many questions are asked of it. The tests in the repository check all
three on an untrained model.

## How it was trained

{training_description}

In training the options were shuffled, long label sets were often shown as a sampled handful, and each
label set was asked through several wordings of its question — so the model reads the options and the
question rather than their positions.

## What it scores

Measured on held-out examples, with **every label offered at once** — all 151 intents of clinc, all 77
of banking — because that is what a request actually looks like. One temperature per label set was
fitted on the same held-out examples. **ECE** is the expected calibration error over 15 equal bands:
how far the stated probability is from how often it turns out right. Rows marked **unseen** are label
sets kept out of training entirely, never trained on, only measured.

{table}

Overall: accuracy **{accuracy}**, log loss **{log_loss}**, Brier **{brier}**, calibration error
**{calibration_error}** (95% interval {interval}).

## The probabilities expire

A temperature is fitted on a distribution, not on a model, so it goes stale when the traffic moves. The
run's `passport.json` records what it was fitted on and what should make you fit it again:

```python
from sharada import check_passport
import json

check_passport(json.load(open("passport.json")), recent_examples, model)
# [] when it finds nothing wrong
```

Fit it again on your own labelled examples before trusting the numbers on your own traffic —
`model.calibrate(examples)` does it in one call, and `model.fit(examples)` does it for you.

## What it will not do

It does not generate: options or nothing. It reads {text_tokens} tokens of text, so longer documents
need chunking. It is English. And it is small — where an answer needs a fact that is not in the text in
front of it, a frontier model wins.

Apache 2.0.
"""


def table(report: dict) -> str:
    rows = ["| label set | options | kind | accuracy | log loss | ECE | T |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    for task, row in sorted(report["per_task"].items(),
                            key=lambda kv: (kv[1]["trained_on"], kv[1]["options"], kv[1]["accuracy"]),
                            reverse=True):
        name = task if row["trained_on"] else f"{task} *(unseen)*"
        rows.append(f"| {name} | {row['options']} | {row['kind']} | {row['accuracy']:.3f} | "
                    f"{row['log_loss']:.3f} | {row['calibration_error']:.3f} | {row['temperature']} |")
    return "\n".join(rows)


def description(report: dict) -> str:
    arguments = report["arguments"]
    unseen = report.get("unseen_label_sets") or []
    kept = report.get("weights_from_step")
    sentence = (f"{report['training_examples']:,} examples from {report['label_sets']} public label sets "
                f"— intents, topics, review scores, emotion, toxicity, spam and entailment — for "
                f"{report['steps']:,} steps of "
                f"{arguments['batch_size'] * arguments.get('accumulate', 1)}, "
                f"AdamW at {arguments['lr']:g} with a cosine schedule, "
                f"{arguments['loss'].replace('_', ' ')} loss. The published weights are the ones that "
                f"measured best on held-out data, at step {kept:,} of {report['steps']:,} — past that "
                f"the model stops answering better and only grows more certain.")
    if unseen:
        sentence += (f" {len(unseen)} further label sets were held out of training entirely and only "
                     f"measured: {', '.join(unseen)}.")
    return sentence


def render(run: pathlib.Path, repo: str) -> str:
    report = json.loads((run / "report.json").read_text())
    overall = report["overall"]
    latency = report.get("latency", {})
    device = report["arguments"].get("device") or ""
    return CARD.format(
        repo=repo,
        encoder=report["encoder"],
        parameters=f"{report['parameters'] / 1e6:.0f}M",
        text_tokens=report["arguments"]["text_tokens"],
        ms_per_question=round(latency.get("ms_per_question", float("nan")), 1),
        device_note=f" on {device}" if device else "",
        training_description=description(report),
        table=table(report),
        accuracy=f"{overall['accuracy']:.3f}",
        log_loss=f"{overall['log_loss']:.3f}",
        brier=f"{overall['brier']:.3f}",
        calibration_error=f"{overall['calibration_error']:.3f}",
        interval="[%.3f, %.3f]" % tuple(overall["calibration_error_interval"]),
    )


def main() -> None:
    parse = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("run", type=pathlib.Path, help="a directory written by training/run.py")
    parse.add_argument("--repo", required=True, help="e.g. LenaBarretta/sharada-base")
    parse.add_argument("--private", action="store_true")
    parse.add_argument("--card-only", action="store_true", help="write the card, upload nothing")
    args = parse.parse_args()

    card = render(args.run, args.repo)
    (args.run / "README.md").write_text(card)
    print(f"wrote {args.run / 'README.md'} ({len(card.splitlines())} lines)")
    if args.card_only:
        return

    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(args.run), repo_id=args.repo, repo_type="model",
                      ignore_patterns=["checkpoint.pt*", "*.writing"])
    print(f"https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
