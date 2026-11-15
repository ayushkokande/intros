"""A minimal LoRA implementation (no peft dependency).

Matches the paper's adapter: rank r on every linear projection of every
transformer layer (q, k, v, o, gate, up, down), rank-stabilised scaling
alpha / sqrt(r).  Each wrapped layer has a `mult` knob so later experiments
can ablate layers or scale adapter contributions (layer ablations,
attribution patching) without reloading weights.
"""

from __future__ import annotations

import json
import math
import re
from contextlib import contextmanager
from pathlib import Path

import torch
from torch import nn

DEFAULT_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Module, r: int, alpha: float, rslora: bool = True):
        super().__init__()
        self.base = base
        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.empty(r, in_f, device=dev, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, device=dev, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / math.sqrt(r) if rslora else alpha / r
        self.mult = 1.0

    def forward(self, x):
        out = self.base(x)
        if self.mult == 0.0:
            return out
        dt = x.dtype
        delta = (x @ self.lora_A.to(dt).t()) @ self.lora_B.to(dt).t()
        return out + delta * (self.scaling * self.mult)


def _layer_of(name: str) -> int | None:
    m = _LAYER_RE.search(name + ".")
    return int(m.group(1)) if m else None


def inject(model: nn.Module, r: int = 8, alpha: float = 8.0, targets=DEFAULT_TARGETS,
           layers: set[int] | None = None, rslora: bool = True) -> dict:
    """Freeze the model and wrap target linears with LoRA.  Returns the config."""
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{name}.{child_name}" if name else child_name
            if child_name not in targets or isinstance(child, LoRALinear):
                continue
            if not hasattr(child, "in_features"):
                continue
            li = _layer_of(full)
            if li is None or (layers is not None and li not in layers):
                continue
            setattr(module, child_name, LoRALinear(child, r, alpha, rslora))
            wrapped += 1
    if wrapped == 0:
        raise RuntimeError("no LoRA targets found")
    cfg = {"r": r, "alpha": alpha, "targets": list(targets), "rslora": rslora,
           "layers": sorted(layers) if layers is not None else None, "n_wrapped": wrapped}
    model._lora_config = cfg
    return cfg


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {n: m for n, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in lora_modules(model).values() for p in (m.lora_A, m.lora_B)]


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    sd = {}
    for n, m in lora_modules(model).items():
        sd[f"{n}.lora_A"] = m.lora_A.detach().cpu().contiguous()
        sd[f"{n}.lora_B"] = m.lora_B.detach().cpu().contiguous()
    return sd


def save(model: nn.Module, path: Path) -> None:
    from safetensors.torch import save_file

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_file(lora_state_dict(model), str(path / "adapter.safetensors"))
    (path / "adapter_config.json").write_text(json.dumps(model._lora_config, indent=1))


def load_into(model: nn.Module, path: Path) -> None:
    """Load adapter weights into an already-injected model (same config)."""
    from safetensors.torch import load_file

    sd = load_file(str(Path(path) / "adapter.safetensors"))
    mods = lora_modules(model)
    missing = set(f"{n}.lora_{x}" for n in mods for x in "AB") - set(sd)
    if missing:
        raise KeyError(f"adapter is missing {len(missing)} tensors, e.g. {sorted(missing)[:2]}")
    with torch.no_grad():
        for n, m in mods.items():
            m.lora_A.copy_(sd[f"{n}.lora_A"].to(m.lora_A.device))
            m.lora_B.copy_(sd[f"{n}.lora_B"].to(m.lora_B.device))


def read_config(path: Path) -> dict:
    return json.loads((Path(path) / "adapter_config.json").read_text())


def set_mult(model: nn.Module, mult: float, layers: set[int] | None = None) -> None:
    for n, m in lora_modules(model).items():
        if layers is None or _layer_of(n) in layers:
            m.mult = mult


@contextmanager
def disabled(model: nn.Module):
    """Run the base model: all adapter contributions off."""
    old = {n: m.mult for n, m in lora_modules(model).items()}
    set_mult(model, 0.0)
    try:
        yield
    finally:
        for n, m in lora_modules(model).items():
            m.mult = old[n]
