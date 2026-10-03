"""Who reads whom.

The text reads only the text, so its states do not depend on the question or the options — a text asked
several questions could be read once. The question reads the text and itself. An option reads the text,
the question and its own tokens, and nothing else: with seventy-seven options on offer, each one still
sees only the message and itself, and its score does not depend on which other options came along.

`read_each_other=True` keeps the other behaviour — options that also read one another — which is what
this model was first trained with. On label sets of a few options it costs nothing; on seventy-seven it
cost thirty points of accuracy, which is why it is not the default.

The encoder's local-attention layers only look a fixed distance away, and the distance is measured in
positions, so an option sees the end of the question the same way wherever it sits in the sequence.
"""

from __future__ import annotations

import torch

from .layout import QUESTION, TEXT


def who_reads_whom(segments: torch.Tensor, positions: torch.Tensor, local_reach: int,
                   read_each_other: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """-> two [batch, length, length] boolean masks: for the global layers and for the local ones.
    True at (i, j) means query token i may read key token j."""
    q, k = segments[:, :, None], segments[:, None, :]
    real = k >= 0                                     # padding is not a key
    context = (k == TEXT) | (k == QUESTION)
    options_see = real if read_each_other else (context | (k == q))
    full = torch.where(q == TEXT, k == TEXT, torch.where(q == QUESTION, context, options_see)) & real
    full = full | torch.eye(segments.size(1), dtype=torch.bool, device=segments.device)[None]
    near = (positions[:, :, None] - positions[:, None, :]).abs() <= local_reach
    return full, full & near


def encoder_masks(segments: torch.Tensor, positions: torch.Tensor, local_reach: int,
                  attn_implementation: str, dtype: torch.dtype, read_each_other: bool = False) -> dict:
    """The masks in the form the encoder expects: booleans for fused attention, an additive float mask
    for the plain implementation."""
    out = {}
    names = ("full_attention", "sliding_attention")
    for name, allowed in zip(names, who_reads_whom(segments, positions, local_reach, read_each_other)):
        mask = allowed[:, None]
        if attn_implementation == "eager":
            mask = torch.zeros(mask.shape, dtype=dtype, device=mask.device).masked_fill(~mask, torch.finfo(dtype).min)
        out[name] = mask
    return out
