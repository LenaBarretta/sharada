"""One request, laid out as one sequence.

    position:  0     1 … n      n+1    n+2 … m     m+1  │ m+2 …         │ m+2 …
    token:     [CLS] the text   [SEP]  question    [SEP]│ option 0 [SEP] │ option 1 [SEP]

Every option is a branch of its own and every branch starts at the same position, the one right after
the question. The encoder's positions are rotary, so to it no option is earlier or later than another:
the order of the options cannot change their scores. `masking.who_reads_whom` then decides who may read
whom; together the two give the properties the model is built on.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

TEXT = 0
QUESTION = 1
FIRST_OPTION = 2        # option i has segment FIRST_OPTION + i


@dataclass(frozen=True)
class Limits:
    """How many tokens each part of a request may use."""

    text: int = 256
    question: int = 48
    option: int = 12

    def as_dict(self) -> dict:
        return {"text": self.text, "question": self.question, "option": self.option}


@dataclass
class Laid:
    """A request turned into token ids, position ids and one segment per token."""

    ids: list[int]
    positions: list[int]
    segments: list[int]
    n_options: int
    option_start: int = field(default=0)

    def __len__(self) -> int:
        return len(self.ids)


class Layout:
    """Lays requests out for one tokenizer. Token ids of repeated strings — option names, instructions,
    the same text asked about twice — are cached, which is most of the cost of preparing a batch."""

    def __init__(self, tokenizer, limits: Limits | None = None):
        self.tokenizer = tokenizer
        self.limits = limits or Limits()
        self._ids = functools.lru_cache(maxsize=100_000)(self._encode)

    def _encode(self, text: str) -> tuple[int, ...]:
        return tuple(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def __call__(self, text: str, question: str, options: list[str]) -> Laid:
        if len(options) < 2:
            raise ValueError("a decision needs at least two options")
        cls_id, sep_id = self.tokenizer.cls_token_id, self.tokenizer.sep_token_id
        body = list(self._ids(text))[: self.limits.text]
        ask = list(self._ids(question))[: self.limits.question]

        ids = [cls_id] + body + [sep_id] + ask + [sep_id]
        segments = [TEXT] * (len(body) + 2) + [QUESTION] * (len(ask) + 1)
        positions = list(range(len(ids)))

        start = len(ids)
        for i, option in enumerate(options):
            tokens = list(self._ids(" " + option))[: self.limits.option] + [sep_id]
            ids += tokens
            segments += [FIRST_OPTION + i] * len(tokens)
            positions += range(start, start + len(tokens))      # every option restarts here
        return Laid(ids=ids, positions=positions, segments=segments, n_options=len(options),
                    option_start=start)
