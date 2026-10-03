"""The decision model: an encoder, a small read-out, one number per option."""

from __future__ import annotations

import json
import pathlib
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn

from .data import Batch, Example, Request, batches, collate, option_mass
from .layout import Laid, Layout, Limits

DEFAULT_ENCODER = "answerdotai/ModernBERT-base"


@dataclass
class Config:
    encoder: str = DEFAULT_ENCODER
    limits: dict = field(default_factory=lambda: Limits().as_dict())
    local_reach: int | None = None          # taken from the encoder when left out
    read_each_other: bool = False           # options reading one another; off by default, see masking.py
    temperatures: dict = field(default_factory=dict)    # per task, fitted by `calibrate`
    version: int = 1


@dataclass
class Decision:
    """What the model answers, and how sure it says it is."""

    answer: str
    index: int
    probabilities: dict[str, float]
    confidence: float

    def __repr__(self) -> str:
        return f"Decision({self.answer!r}, confidence={self.confidence:.3f})"


class DecisionModel(nn.Module):
    def __init__(self, config: Config | None = None, tokenizer=None, encoder=None):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.config = config or Config()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(self.config.encoder)
        self.attn = "eager" if not torch.cuda.is_available() else "sdpa"
        self.encoder = encoder or AutoModel.from_pretrained(self.config.encoder, attn_implementation=self.attn)
        hidden = self.encoder.config.hidden_size
        self.read = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(), nn.LayerNorm(hidden),
                                  nn.Linear(hidden, 1))
        if self.config.local_reach is None:
            self.config.local_reach = getattr(self.encoder.config, "local_attention", 128) // 2
        self.layout = Layout(self.tokenizer, Limits(**self.config.limits))

    # ── the forward pass ──────────────────────────────────────────────────────────────────────
    def forward(self, batch: Batch) -> torch.Tensor:
        """-> [batch, options] scores; options that do not exist cannot win."""
        from .masking import encoder_masks

        dtype = self.encoder.embeddings.tok_embeddings.weight.dtype
        masks = encoder_masks(batch.segments, batch.positions, self.config.local_reach,
                              self.attn, dtype, self.config.read_each_other)
        states = self.encoder(input_ids=batch.ids, position_ids=batch.positions,
                              attention_mask=masks).last_hidden_state
        options = option_mass(batch.segments, states, batch.width)
        summary = states[:, :1].expand(-1, batch.width, -1)
        scores = self.read(torch.cat([options, options * summary], -1)).squeeze(-1).float()
        return scores.masked_fill(~batch.present, -1e4)

    # ── asking it things ──────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def probabilities(self, requests: list[Request], batch_size: int = 32,
                      calibrated: bool = True) -> list[list[float]]:
        """One distribution over the options of each request, in the order they were given."""
        self.eval()
        device = next(self.parameters()).device
        out: list[list[float]] = [[] for _ in requests]
        for chunk, batch in batches(requests, self.layout, batch_size, self.tokenizer.pad_token_id):
            scores = self(batch.to(device))
            for row, j in enumerate(chunk):
                k = int(batch.n_options[row])
                z = scores[row, :k].float().cpu()
                if calibrated:
                    z = z / self.temperature_for(requests[j])
                out[j] = torch.softmax(z, -1).tolist()
        return out

    def decide(self, text: str, question: str, options: list[str], kind: str = "choice",
               task: str = "task", calibrated: bool = True) -> Decision:
        """One decision, start to finish."""
        request = Request(text=text, question=question, options=options, kind=kind, task=task)
        return self.decide_many([request], calibrated=calibrated)[0]

    def decide_many(self, requests: list[Request], batch_size: int = 32,
                    calibrated: bool = True) -> list[Decision]:
        decisions = []
        for request, probabilities in zip(requests, self.probabilities(requests, batch_size, calibrated)):
            best = max(range(len(probabilities)), key=probabilities.__getitem__)
            decisions.append(Decision(answer=request.options[best], index=best,
                                      probabilities=dict(zip(request.options, probabilities)),
                                      confidence=probabilities[best]))
        return decisions

    def temperature_for(self, request: Request) -> float:
        return float(self.config.temperatures.get(request.task, 1.0))

    # ── saving and loading ────────────────────────────────────────────────────────────────────
    def save(self, path: str | pathlib.Path) -> pathlib.Path:
        from safetensors.torch import save_file

        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.state_dict().items()}, path / "model.safetensors")
        (path / "config.json").write_text(json.dumps(asdict(self.config), indent=1))
        self.tokenizer.save_pretrained(path)
        return path

    @classmethod
    def from_pretrained(cls, path: str | pathlib.Path, device: str | None = None) -> "DecisionModel":
        """A local directory saved by `save`, or a model id on the Hub."""
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        local = pathlib.Path(path)
        if not local.is_dir():
            from huggingface_hub import snapshot_download

            local = pathlib.Path(snapshot_download(str(path)))
        config = Config(**json.loads((local / "config.json").read_text()))
        tokenizer = AutoTokenizer.from_pretrained(local)
        attn = "eager" if not torch.cuda.is_available() else "sdpa"
        encoder = AutoModel.from_config(AutoConfig.from_pretrained(config.encoder), attn_implementation=attn)
        model = cls(config=config, tokenizer=tokenizer, encoder=encoder)
        model.load_state_dict(load_file(local / "model.safetensors"))
        return model.to(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def push_to_hub(self, repo_id: str, private: bool = False) -> str:
        """Upload to the Hub. The token comes from the environment; this never asks for one."""
        import tempfile

        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(repo_id, private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            self.save(tmp)
            api.upload_folder(folder_path=tmp, repo_id=repo_id)
        return f"https://huggingface.co/{repo_id}"

    # ── training lives in train.py, calibration in calibrate.py ───────────────────────────────
    def fit(self, examples: list[Example], **kwargs):
        from .train import fit

        return fit(self, examples, **kwargs)

    def calibrate(self, examples: list[Example], **kwargs):
        from .calibrate import calibrate

        return calibrate(self, examples, **kwargs)


def encode_one(model: DecisionModel, request: Request) -> tuple[Laid, Batch]:
    """Handy in tests: one request, laid out and collated."""
    laid = model.layout(request.text, request.question, request.options)
    return laid, collate([laid], [request], model.tokenizer.pad_token_id)
