"""Fine-tune on your own labelled rows.

    python examples/finetune_your_own.py --csv examples/tickets.csv \
        --question "Which team should handle this?" --out my-router

The CSV needs a `text` column and a `label` column holding the option as written. The options are read
off the file, in the order they first appear — for a `scale` pass --options to fix the order instead.

Twenty-four rows, as in the sample file, is enough to see the pipeline work and not enough to trust the
result; a few hundred per option is the point where this gets useful. With `--freeze-encoder` only the
read-out trains, which takes seconds and is often the better choice when the rows are few.
"""

from __future__ import annotations

import argparse
import csv
import pathlib

from sharada import DecisionModel, Example, check_passport, save_passport


def load(path: pathlib.Path, question: str, options: list[str] | None,
         kind: str, task: str) -> list[Example]:
    rows = list(csv.DictReader(path.open()))
    if not rows or "text" not in rows[0] or "label" not in rows[0]:
        raise SystemExit(f"{path} needs a header with a `text` and a `label` column")
    if options is None:
        options = list(dict.fromkeys(row["label"].strip() for row in rows))
    if len(options) < 2:
        raise SystemExit("found one label in the file; a decision needs at least two options")
    index = {option: i for i, option in enumerate(options)}

    examples = []
    for n, row in enumerate(rows, start=2):
        label = row["label"].strip()
        if label not in index:
            raise SystemExit(f"{path}:{n}: label {label!r} is not one of {options}")
        examples.append(Example(text=row["text"].strip(), question=question, options=options,
                                kind=kind, task=task, label=index[label]))
    return examples


def main() -> None:
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("examples/tickets.csv"))
    parse.add_argument("--question", default="Which team should handle this?")
    parse.add_argument("--options", nargs="+", help="fix the options and their order")
    parse.add_argument("--kind", default="choice", choices=["choice", "scale", "binary"])
    parse.add_argument("--task", default="my-task", help="the name the temperature is fitted under")
    parse.add_argument("--model", default="LenaBarretta/sharada-base")
    parse.add_argument("--out", type=pathlib.Path, default=pathlib.Path("my-router"))
    parse.add_argument("--freeze-encoder", action="store_true", help="train the read-out only")
    parse.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "brier"])
    parse.add_argument("--epochs", type=int, default=10)
    args = parse.parse_args()

    examples = load(args.csv, args.question, args.options, args.kind, args.task)
    print(f"{len(examples)} examples, {len(examples[0].options)} options: {examples[0].options}")

    model = DecisionModel.from_pretrained(args.model)
    before = model.decide(examples[0].text, args.question, examples[0].options,
                          kind=args.kind, task=args.task)
    print(f"before: {before}")

    report = model.fit(examples, loss=args.loss, max_epochs=args.epochs,
                       freeze_encoder=args.freeze_encoder)
    print(f"\n{report}")
    print(f"temperatures: {report.temperatures}")

    after = model.decide(examples[0].text, args.question, examples[0].options,
                         kind=args.kind, task=args.task)
    print(f"after:  {after}")

    model.save(args.out)
    save_passport(report.passport, args.out / "passport.json")
    print(f"\nsaved to {args.out}/  (model.safetensors, config.json, tokenizer, passport.json)")

    problems = check_passport(report.passport, examples, model)
    for problem in problems:
        print(f"  ! {problem}")
    if not problems:
        print("  passport clean")

    print("\nto use it:  DecisionModel.from_pretrained(%r)" % str(args.out))
    print("to publish: model.push_to_hub('your-name/your-router')   # HF_TOKEN in the environment")


if __name__ == "__main__":
    main()
