"""Measuring a decision model: the answer, the stated number, and the whole distribution."""

from __future__ import annotations

import math
import random
import time

from .data import Example


def _bands(confidence: list[float], n: int) -> list[int]:
    return [min(int(c * n), n - 1) for c in confidence]


def calibration_error(confidence: list[float], correct: list[int], bands: int = 15) -> float:
    """The average gap between the probability stated and how often it turns out right, over equal
    bands, weighted by how many answers fall in each. Zero means this estimate finds no gap — on a
    finite sample that is not proof of none."""
    index = _bands(confidence, bands)
    total = 0.0
    for b in range(bands):
        inside = [i for i, j in enumerate(index) if j == b]
        if inside:
            said = sum(confidence[i] for i in inside) / len(inside)
            was = sum(correct[i] for i in inside) / len(inside)
            total += len(inside) / len(confidence) * abs(said - was)
    return total


def calibration_interval(confidence: list[float], correct: list[int], bands: int = 15,
                         draws: int = 200, seed: int = 0) -> tuple[float, float]:
    """A bootstrap interval for the calibration error — on a few hundred answers it is wide, and
    saying so is the point."""
    rng = random.Random(seed)
    n = len(confidence)
    values = []
    for _ in range(draws):
        pick = [rng.randrange(n) for _ in range(n)]
        values.append(calibration_error([confidence[i] for i in pick], [correct[i] for i in pick], bands))
    values.sort()
    return values[int(0.025 * draws)], values[int(0.975 * draws) - 1]


def reliability_curve(confidence: list[float], correct: list[int], bands: int = 10) -> list[dict]:
    index = _bands(confidence, bands)
    curve = []
    for b in range(bands):
        inside = [i for i, j in enumerate(index) if j == b]
        curve.append({"from": b / bands, "n": len(inside),
                      "stated": sum(confidence[i] for i in inside) / len(inside) if inside else None,
                      "observed": sum(correct[i] for i in inside) / len(inside) if inside else None})
    return curve


def risk_coverage(confidence: list[float], correct: list[int], steps: int = 20) -> list[dict]:
    """Answer only where the model is surest: what accuracy, at what share of the traffic."""
    order = sorted(range(len(confidence)), key=lambda i: -confidence[i])
    out = []
    for k in range(1, steps + 1):
        take = max(1, round(len(order) * k / steps))
        kept = [correct[i] for i in order[:take]]
        out.append({"coverage": take / len(order), "accuracy": sum(kept) / len(kept),
                    "threshold": confidence[order[take - 1]]})
    return out


def evaluate(model, examples: list[Example], batch_size: int = 32, calibrated: bool = True) -> dict:
    """Everything above, on one set of labelled examples."""
    probabilities = model.probabilities(examples, batch_size=batch_size, calibrated=calibrated)
    confidence, correct, true_p, squared = [], [], [], []
    for p, e in zip(probabilities, examples):
        best = max(range(len(p)), key=p.__getitem__)
        confidence.append(p[best])
        correct.append(int(best == e.label))
        true_p.append(p[e.label])
        squared.append(sum((q - (1.0 if i == e.label else 0.0)) ** 2 for i, q in enumerate(p)))
    low, high = calibration_interval(confidence, correct)
    return {"n": len(examples),
            "accuracy": sum(correct) / len(correct),
            "mean_confidence": sum(confidence) / len(confidence),
            "calibration_error": calibration_error(confidence, correct),
            "calibration_error_interval": [low, high],
            "log_loss": sum(-math.log(max(p, 1e-9)) for p in true_p) / len(true_p),
            "brier": sum(squared) / len(squared),
            "reliability": reliability_curve(confidence, correct),
            "risk_coverage": risk_coverage(confidence, correct)}


def latency(model, request, batch_size: int = 1, repeats: int = 20) -> dict:
    """Milliseconds per call and per question, as the model is loaded right now."""
    import torch

    requests = [request] * batch_size
    for _ in range(3):
        model.probabilities(requests, batch_size=batch_size)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        model.probabilities(requests, batch_size=batch_size)
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()
    per_call = (time.time() - start) / repeats * 1e3
    return {"batch": batch_size, "ms_per_call": per_call, "ms_per_question": per_call / batch_size}
