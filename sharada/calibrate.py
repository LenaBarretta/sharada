"""Calibration, and the passport that says where it holds.

A model is not calibrated in general; it is calibrated on a distribution, and it goes stale when the
traffic moves. So fitting a temperature also writes a passport: what it was fitted on, how far off the
probabilities were, and what should make you fit it again.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import pathlib

from .data import Example

VALID_DAYS = 90


def _negative_log_likelihood(log_probabilities: list[list[float]], labels: list[int], temperature: float) -> float:
    total = 0.0
    for row, label in zip(log_probabilities, labels):
        scaled = [v / temperature for v in row]
        top = max(scaled)
        total += -(scaled[label] - (top + math.log(sum(math.exp(v - top) for v in scaled))))
    return total / len(labels)


def fit_temperature(probabilities: list[list[float]], labels: list[int],
                    grid: tuple[float, float, int] = (0.25, 8.0, 61)) -> float:
    """One number per task: divide the scores by it. Chosen to minimise the log loss on held-out
    answers, which is what a temperature can honestly be chosen on."""
    low, high, steps = grid
    log_probabilities = [[math.log(max(p, 1e-12)) for p in row] for row in probabilities]
    candidates = [math.exp(math.log(low) + (math.log(high) - math.log(low)) * i / (steps - 1)) for i in range(steps)]
    return min(candidates, key=lambda t: _negative_log_likelihood(log_probabilities, labels, t))


def fingerprint(model, examples: list[Example]) -> dict:
    """What the examples looked like, so a later drift is visible."""
    out: dict[str, dict] = {}
    for task in sorted({e.task for e in examples}):
        subset = [e for e in examples if e.task == task]
        lengths = sorted(len(model.layout(e.text, e.question, e.options).ids) for e in subset)
        counts: dict[str, int] = {}
        for e in subset:
            counts[e.options[e.label]] = counts.get(e.options[e.label], 0) + 1
        out[task] = {"n": len(subset), "options": subset[0].options,
                     "median_tokens": lengths[len(lengths) // 2],
                     "answers": counts}
    return out


def calibrate(model, examples: list[Example], batch_size: int = 32, valid_days: int = VALID_DAYS) -> dict:
    """Fit one temperature per task on these examples and return the passport. The temperatures are
    stored on the model, so `decide` uses them from then on."""
    from .evaluate import evaluate

    raw = model.probabilities(examples, batch_size=batch_size, calibrated=False)
    for task in sorted({e.task for e in examples}):
        index = [i for i, e in enumerate(examples) if e.task == task]
        model.config.temperatures[task] = fit_temperature([raw[i] for i in index],
                                                          [examples[i].label for i in index])
    measured = evaluate(model, examples, batch_size=batch_size, calibrated=True)
    today = dt.date.today()
    return {
        "created": today.isoformat(),
        "valid_until": (today + dt.timedelta(days=valid_days)).isoformat(),
        "encoder": model.config.encoder,
        "examples": len(examples),
        "temperatures": dict(model.config.temperatures),
        "fingerprint": fingerprint(model, examples),
        "calibration_error": measured["calibration_error"],
        "calibration_error_interval": measured["calibration_error_interval"],
        "accuracy": measured["accuracy"],
        "reliability": measured["reliability"],
        "risk_coverage": measured["risk_coverage"],
        "recalibrate_if": {
            "after": "valid_until",
            "options_change": True,
            "median_tokens_shift": 0.5,       # half as long, or twice as long
        },
    }


def save_passport(passport: dict, path: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(path)
    path.write_text(json.dumps(passport, indent=1))
    return path


def check_passport(passport: dict, examples: list[Example] | None = None, model=None,
                   today: dt.date | None = None) -> list[str]:
    """What is wrong with trusting these probabilities today. An empty list means nothing found —
    run it in the deployment pipeline and fail the build on anything it returns."""
    today = today or dt.date.today()
    problems = []
    if today.isoformat() > passport["valid_until"]:
        problems.append(f"the calibration expired on {passport['valid_until']}; fit it again on recent answers")
    if passport["examples"] < 200:
        problems.append(f"fitted on {passport['examples']} answers — the interval on the calibration error is wide")
    if examples and model:
        now = fingerprint(model, examples)
        for task, then in passport["fingerprint"].items():
            if task not in now:
                continue
            if now[task]["options"] != then["options"]:
                problems.append(f"{task}: the options have changed since the calibration")
            ratio = now[task]["median_tokens"] / max(then["median_tokens"], 1)
            if ratio < 0.5 or ratio > 2:
                problems.append(f"{task}: texts are now {ratio:.1f}× the length they were calibrated on")
        for task in now:
            if task not in passport["fingerprint"]:
                problems.append(f"{task}: never calibrated — its probabilities are raw")
    return problems
