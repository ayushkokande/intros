"""A small GPT-2 built from a config (random init), plus save/load and LR schedule."""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def make_model(vocab: int, n_layer: int, n_embd: int, n_head: int, dropout: float = 0.0,
               n_positions: int = 64, seed: int = 0):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=vocab, n_positions=n_positions, n_embd=n_embd, n_layer=n_layer,
                     n_head=n_head, resid_pdrop=dropout, embd_pdrop=dropout, attn_pdrop=dropout,
                     bos_token_id=1, eos_token_id=2, pad_token_id=0,
                     # Untied, so character rows of the input embedding are only ever
                     # changed by appearing in the input (never as output negatives).
                     tie_word_embeddings=False)
    torch.manual_seed(seed)
    return GPT2LMHeadModel(cfg)


def save(model, path: Path, extra: dict | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if extra:
        (path / "extra.json").write_text(json.dumps(extra, indent=1))


def load(path: Path, device):
    from transformers import GPT2LMHeadModel

    return GPT2LMHeadModel.from_pretrained(path).to(device)


def device_of(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def lm_loss(model, x, y, attn):
    """Next-token loss on positions where y != -100.  Returns (loss, logits)."""
    logits = model(input_ids=x, attention_mask=attn).logits
    loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                           y[:, 1:].reshape(-1), ignore_index=-100)
    return loss, logits


def lr_lambda(warmup: int, steps: int, floor: float = 0.1):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = min(1.0, (step - warmup) / max(1, steps - warmup))
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog))
    return f
