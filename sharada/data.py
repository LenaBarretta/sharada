"""Requests, examples and batches."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .layout import FIRST_OPTION, Laid, Layout

KINDS = ("choice", "scale", "binary")       # unordered labels, an ordered scale, yes or no


@dataclass
class Request:
    """One question about one text."""

    text: str
    question: str
    options: list[str]
    kind: str = "choice"
    task: str = "task"                       # a name, so temperatures can be fitted per task

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")


@dataclass
class Example(Request):
    """A request whose answer is known: `label` is the index of the right option."""

    label: int = 0

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.label < len(self.options):
            raise ValueError(f"label {self.label} is outside the {len(self.options)} options")


@dataclass
class Batch:
    ids: torch.Tensor
    positions: torch.Tensor
    segments: torch.Tensor
    present: torch.Tensor                    # [batch, options] — which option slots are real
    kinds: torch.Tensor
    n_options: torch.Tensor
    labels: torch.Tensor | None = None
    tasks: list[str] = field(default_factory=list)
    width: int = 0                           # the widest number of options in this batch

    def to(self, device) -> "Batch":
        move = lambda v: v.to(device) if torch.is_tensor(v) else v
        return Batch(move(self.ids), move(self.positions), move(self.segments), move(self.present),
                     move(self.kinds), move(self.n_options), move(self.labels), self.tasks, self.width)


def collate(laid: list[Laid], requests: list[Request], pad_id: int) -> Batch:
    length = max(len(l) for l in laid)
    width = max(l.n_options for l in laid)
    n = len(laid)
    ids = torch.full((n, length), pad_id, dtype=torch.long)
    positions = torch.zeros((n, length), dtype=torch.long)
    segments = torch.full((n, length), -1, dtype=torch.long)
    for i, l in enumerate(laid):
        k = len(l)
        ids[i, :k] = torch.tensor(l.ids)
        positions[i, :k] = torch.tensor(l.positions)
        segments[i, :k] = torch.tensor(l.segments)
    counts = torch.tensor([l.n_options for l in laid])
    labels = None
    if all(isinstance(r, Example) for r in requests):
        labels = torch.tensor([r.label for r in requests])
    return Batch(ids=ids, positions=positions, segments=segments,
                 present=torch.arange(width)[None] < counts[:, None],
                 kinds=torch.tensor([KINDS.index(r.kind) for r in requests]),
                 n_options=counts, labels=labels, tasks=[r.task for r in requests], width=width)


def batches(requests: list[Request], layout: Layout, batch_size: int, pad_id: int,
            shuffle: bool = False, seed: int = 0):
    """Mini-batches. When shuffling, requests of similar length travel together, so a batch of short
    questions is not padded out to the length of one with sixty options."""
    order = list(range(len(requests)))
    laid = [layout(r.text, r.question, r.options) for r in requests]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(order)
        window = batch_size * 50
        order = [j for w in range(0, len(order), window)
                 for j in sorted(order[w:w + window], key=lambda j: len(laid[j]))]
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        random.Random(seed + 1).shuffle(chunks)
    for chunk in chunks:
        yield chunk, collate([laid[j] for j in chunk], [requests[j] for j in chunk], pad_id)


def option_mass(segments: torch.Tensor, states: torch.Tensor, width: int) -> torch.Tensor:
    """Mean of each option's own final token states -> [batch, options, hidden]."""
    index = FIRST_OPTION + torch.arange(width, device=states.device)
    member = (segments[:, :, None] == index[None, None, :]).to(states.dtype)
    return torch.einsum("blk,bld->bkd", member, states) / member.sum(1).clamp(min=1)[..., None]
