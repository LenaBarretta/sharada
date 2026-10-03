"""Fine-tuning on your own labelled examples.

Three lines should get you a usable model:

    model = DecisionModel.from_pretrained("LenaBarretta/sharada-base")
    report = model.fit(examples)
    model.save("my-router")

`fit` holds a part of the examples out, trains on the rest, stops when the held-out log loss stops
improving, then fits one temperature per task on that same held-out part and writes a calibration
passport. What comes back is a report of what it did and how well it ended up.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import Example, batches


@dataclass
class Report:
    examples: int
    held_out: int
    epochs: float
    minutes: float
    accuracy: float
    log_loss: float
    calibration_error: float
    temperatures: dict
    passport: dict

    def __repr__(self) -> str:
        return (f"Report({self.examples} examples, accuracy {self.accuracy:.3f}, "
                f"log loss {self.log_loss:.3f}, calibration error {self.calibration_error:.3f})")


def brier_loss(scores: torch.Tensor, batch) -> torch.Tensor:
    """The squared error over all options; for a scale, over the cumulative distribution."""
    probabilities = torch.softmax(scores, -1) * batch.present
    truth = F.one_hot(batch.labels, scores.size(-1)).to(probabilities.dtype) * batch.present
    flat = ((probabilities - truth) ** 2 * batch.present).sum(-1)
    ranked = (((probabilities.cumsum(-1) - truth.cumsum(-1)) ** 2) * batch.present).sum(-1) / (batch.n_options - 1).clamp(min=1)
    return torch.where(batch.kinds == 1, ranked, flat).mean()


def fit(model, examples: list[Example], *, val_split: float = 0.2, max_epochs: int = 10,
        batch_size: int = 16, lr: float = 2e-5, patience: int = 2, loss: str = "cross_entropy",
        freeze_encoder: bool = False, seed: int = 0, device: str | None = None,
        progress: bool = True) -> Report:
    if len(examples) < 20:
        raise ValueError("fine-tuning needs at least 20 examples; below that, use the model as it is")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rng = random.Random(seed)
    order = list(range(len(examples)))
    rng.shuffle(order)
    cut = max(10, int(len(examples) * val_split))
    held_out = [examples[i] for i in order[:cut]]
    training = [examples[i] for i in order[cut:]]

    for parameter in model.encoder.parameters():
        parameter.requires_grad = not freeze_encoder
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    steps = max_epochs * math.ceil(len(training) / batch_size)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps + 1, pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    loss_fn = brier_loss if loss == "brier" else None

    best, best_state, waited, start, epochs_done = math.inf, None, 0, time.time(), 0
    for epoch in range(max_epochs):
        model.train()
        for _, batch in batches(training, model.layout, batch_size, model.tokenizer.pad_token_id,
                                shuffle=True, seed=seed + epoch):
            batch = batch.to(device)
            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                scores = model(batch)
                step_loss = loss_fn(scores, batch) if loss_fn else F.cross_entropy(scores, batch.labels)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(step_loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
        epochs_done = epoch + 1

        current = _log_loss(model, held_out, batch_size)
        if progress:
            print(f"epoch {epochs_done}: held-out log loss {current:.4f}", flush=True)
        if current < best - 1e-4:
            best, waited = current, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    from .calibrate import calibrate

    passport = calibrate(model, held_out, batch_size=batch_size)
    from .evaluate import evaluate

    scores = evaluate(model, held_out, batch_size=batch_size)
    return Report(examples=len(examples), held_out=len(held_out), epochs=epochs_done,
                  minutes=(time.time() - start) / 60, accuracy=scores["accuracy"],
                  log_loss=scores["log_loss"], calibration_error=scores["calibration_error"],
                  temperatures=dict(model.config.temperatures), passport=passport)


def _log_loss(model, examples: list[Example], batch_size: int) -> float:
    probabilities = model.probabilities(examples, batch_size=batch_size, calibrated=False)
    return float(sum(-math.log(max(p[e.label], 1e-9)) for p, e in zip(probabilities, examples)) / len(examples))
