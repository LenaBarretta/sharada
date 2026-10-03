"""Train a base model on the whole mix.

    python training/run.py --encoder answerdotai/ModernBERT-base --out runs/base
    python training/run.py --encoder answerdotai/ModernBERT-large --out runs/large --batch-size 8

This is the pretraining run, not the user-facing one: `sharada.fit` is for a few hundred of your own
examples, while this reads thirty-odd public label sets and walks over them for a fixed number of steps.
It is a step loop rather than an epoch loop because the mix is large and the budget is a session on a
free GPU.

It writes a checkpoint every `--checkpoint-every` steps and, when `--out` already holds one, carries on
from it — the step counter, the optimiser and the schedule included. A Kaggle session that runs out of
time is then a restart, not a loss. Anything already finished is kept, so re-running after a full run
only re-measures.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from sharada import Config, DecisionModel, calibrate, evaluate, latency, save_passport      # noqa: E402
from sharada.data import batches                                                            # noqa: E402
from sharada.train import brier_loss                                                        # noqa: E402

import sources                                                                              # noqa: E402

CHECKPOINT = "checkpoint.pt"


def arguments(argv=None):
    parse = argparse.ArgumentParser(description=__doc__,
                                    formatter_class=argparse.RawDescriptionHelpFormatter)
    parse.add_argument("--encoder", default="answerdotai/ModernBERT-base")
    parse.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs/base"))
    parse.add_argument("--steps", type=int, default=12_000)
    parse.add_argument("--batch-size", type=int, default=16)
    parse.add_argument("--lr", type=float, default=3e-5)
    parse.add_argument("--warmup", type=float, default=0.06, help="share of the steps spent warming up")
    parse.add_argument("--loss", default="cross_entropy", choices=["cross_entropy", "brier"])
    parse.add_argument("--text-tokens", type=int, default=256)
    parse.add_argument("--cap-scale", type=float, default=1.0, help="scale every source's cap")
    parse.add_argument("--eval-every", type=int, default=1000)
    parse.add_argument("--eval-examples", type=int, default=250, help="per label set, when measuring")
    parse.add_argument("--checkpoint-every", type=int, default=500)
    parse.add_argument("--seed", type=int, default=0)
    parse.add_argument("--device", default=None)
    parse.add_argument("--push-to", default=None, help="Hub id to upload to when it is done")
    parse.add_argument("--fresh", action="store_true", help="ignore any checkpoint in --out")
    parse.add_argument("--only", nargs="+", default=None,
                       help="train on these label sets only, by task name (for trying the run out)")
    return parse.parse_args(argv)


def say(*parts) -> None:
    print(*parts, flush=True)


# ── the data ─────────────────────────────────────────────────────────────────────────────────────

def mixture(args):
    chosen = sources.SOURCES
    if args.only:
        chosen = tuple(s for s in chosen if s.task in set(args.only))
        missing = set(args.only) - {s.task for s in chosen}
        if missing:
            raise SystemExit(f"no such label set: {sorted(missing)}")
    say("training mix:")
    training = sources.build("train", seed=args.seed, cap_scale=args.cap_scale,
                             sources=chosen, log=say)
    say("\nmeasured on (the held-out part of each source, plus the label sets kept out of training):")
    held_out = sources.build("eval", seed=args.seed, cap_scale=args.eval_examples / 2500,
                             include_holdout=True, sources=chosen, log=say)
    return training, held_out


# ── the loop ─────────────────────────────────────────────────────────────────────────────────────

def learning_rate(step: int, total: int, peak: float, warmup: float) -> float:
    warm = max(1, int(total * warmup))
    if step < warm:
        return peak * step / warm
    done = (step - warm) / max(1, total - warm)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(done, 1.0))))


def forever(examples, model, batch_size, seed):
    """Batches, for as long as they are asked for, reshuffled each time round."""
    epoch = 0
    while True:
        for _, batch in batches(examples, model.layout, batch_size, model.tokenizer.pad_token_id,
                                shuffle=True, seed=seed + epoch):
            yield batch
        epoch += 1


def train(args, model, training, held_out, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                                  betas=(0.9, 0.98), eps=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    loss_fn = brier_loss if args.loss == "brier" else None

    state, done = _resume(args, model, optimizer, scaler, device)
    history = state.get("history", [])
    start = state.get("step", 0)
    if done:
        return history, start

    stream = forever(training, model, args.batch_size, args.seed + start)
    model.train()
    running, began = [], time.time()
    for step in range(start + 1, args.steps + 1):
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.steps, args.lr, args.warmup)
        batch = next(stream).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
            scores = model(batch)
            loss = loss_fn(scores, batch) if loss_fn else F.cross_entropy(scores, batch.labels)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running.append(float(loss))

        if step % 100 == 0:
            rate = (time.time() - began) / step_count(step, start)
            say(f"step {step:6d}/{args.steps}  loss {sum(running) / len(running):.4f}  "
                f"lr {optimizer.param_groups[0]['lr']:.2e}  {rate:.2f}s/step  "
                f"eta {(args.steps - step) * rate / 60:.0f} min")
            running = []
        if step % args.eval_every == 0 or step == args.steps:
            measured = measure(model, held_out, args)
            history.append({"step": step, **measured})
            say(f"  step {step}: accuracy {measured['accuracy']:.3f}  log loss {measured['log_loss']:.3f}"
                f"  calibration error {measured['calibration_error']:.3f}"
                f"  (unseen label sets: {measured['holdout_accuracy']:.3f})")
            model.train()
        if step % args.checkpoint_every == 0 or step == args.steps:
            _save_checkpoint(args, model, optimizer, scaler, step, history)
    return history, args.steps


def step_count(step: int, start: int) -> int:
    return max(1, step - start)


def _save_checkpoint(args, model, optimizer, scaler, step, history) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    tmp = args.out / (CHECKPOINT + ".writing")
    torch.save({"step": step, "history": history, "arguments": vars(args) | {"out": str(args.out)},
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()}, tmp)
    tmp.replace(args.out / CHECKPOINT)      # one atomic move: a killed session cannot leave half a file


def _resume(args, model, optimizer, scaler, device) -> tuple[dict, bool]:
    path = args.out / CHECKPOINT
    if args.fresh or not path.exists():
        return {}, False
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scaler.load_state_dict(state["scaler"])
    say(f"\nresuming from {path} at step {state['step']} of {args.steps}")
    return state, state["step"] >= args.steps


# ── measuring ────────────────────────────────────────────────────────────────────────────────────

def measure(model, held_out, args) -> dict:
    overall = evaluate(model, held_out, batch_size=max(32, args.batch_size), calibrated=False)
    unseen = {s.task for s in sources.holdout_sources()}
    only_unseen = [e for e in held_out if e.task in unseen]
    holdout = evaluate(model, only_unseen, batch_size=max(32, args.batch_size),
                       calibrated=False) if only_unseen else {"accuracy": float("nan")}
    return {"accuracy": overall["accuracy"], "log_loss": overall["log_loss"],
            "brier": overall["brier"], "calibration_error": overall["calibration_error"],
            "holdout_accuracy": holdout["accuracy"]}


def per_task(model, held_out, args) -> dict:
    out = {}
    unseen = {s.task for s in sources.holdout_sources()}
    for task in sorted({e.task for e in held_out}):
        subset = [e for e in held_out if e.task == task]
        scored = evaluate(model, subset, batch_size=max(32, args.batch_size), calibrated=True)
        out[task] = {"n": scored["n"], "options": len(subset[0].options), "kind": subset[0].kind,
                     "trained_on": task not in unseen,
                     "accuracy": round(scored["accuracy"], 4),
                     "log_loss": round(scored["log_loss"], 4),
                     "brier": round(scored["brier"], 4),
                     "calibration_error": round(scored["calibration_error"], 4),
                     "calibration_error_interval": [round(v, 4) for v in scored["calibration_error_interval"]],
                     "temperature": round(model.temperature_for(subset[0]), 3)}
    return out


# ── putting it together ──────────────────────────────────────────────────────────────────────────

def main(argv=None) -> None:
    args = arguments(argv)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    say(f"{args.encoder} -> {args.out}  on {device}, {args.steps} steps of {args.batch_size}")

    training, held_out = mixture(args)
    model = DecisionModel(Config(encoder=args.encoder,
                                 limits={"text": args.text_tokens, "question": 48, "option": 12}))
    say(f"\n{sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    model.to(device)

    history, step = train(args, model, training, held_out, device)

    say("\nfitting one temperature per label set on the held-out examples")
    passport = calibrate(model, held_out, batch_size=max(32, args.batch_size))
    scores = evaluate(model, held_out, batch_size=max(32, args.batch_size))
    tasks = per_task(model, held_out, args)

    model.save(args.out)
    save_passport(passport, args.out / "passport.json")
    report = {"encoder": args.encoder, "steps": step, "arguments": vars(args) | {"out": str(args.out)},
              "parameters": sum(p.numel() for p in model.parameters()),
              "training_examples": len(training), "held_out_examples": len(held_out),
              "label_sets": len({e.task for e in training}),
              "unseen_label_sets": sorted({e.task for e in held_out} & {s.task for s in sources.holdout_sources()}),
              "overall": {k: scores[k] for k in ("accuracy", "log_loss", "brier",
                                                 "calibration_error", "calibration_error_interval")},
              "per_task": tasks, "history": history,
              "latency": latency(model, held_out[0]),
              "sources": sources.catalogue()}
    (args.out / "report.json").write_text(json.dumps(report, indent=1))

    say(f"\n{args.out}: accuracy {scores['accuracy']:.3f}, log loss {scores['log_loss']:.3f}, "
        f"calibration error {scores['calibration_error']:.3f}")
    for task, row in tasks.items():
        say(f"  {'   ' if row['trained_on'] else 'new'} {task:18s} {row['options']:3d} options  "
            f"accuracy {row['accuracy']:.3f}  ece {row['calibration_error']:.3f}  T {row['temperature']}")

    if args.push_to:
        say(f"\nuploading to {args.push_to}")
        say(model.push_to_hub(args.push_to))


if __name__ == "__main__":
    main()
