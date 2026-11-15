"""Model loading, chat rendering, batched A/B scoring and batched generation."""

from __future__ import annotations

import torch

from . import prompts


def load(model_name: str, quant: str = "none", device_map: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"dtype": torch.bfloat16}
    if quant in ("8bit", "4bit"):
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if quant == "8bit"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_quant_type="nf4")
        )
    if torch.cuda.is_available():
        kwargs["device_map"] = device_map
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.config.use_cache = False
    return model, tok


def input_device(model) -> torch.device:
    return model.get_input_embeddings().weight.device


def render(tok, user: str, system: str = prompts.SYSTEM) -> str:
    """Chat-format a prompt, ending with the assistant turn in non-reasoning mode.

    For Qwen3 `enable_thinking=False` appends the empty <think></think> prefill
    used in the paper; other templates ignore the flag.
    """
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)


def token_ids(tok, words) -> list[int]:
    """Single-token ids for each word (raises if a word is not one token)."""
    ids = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) != 1:
            raise ValueError(f"{w!r} is not a single token: {t}")
        ids.append(t[0])
    return ids


def letter_ids(tok) -> tuple[int, int]:
    ia, ib = token_ids(tok, "AB")
    return ia, ib


def encode(tok, texts: list[str], device) -> dict:
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    return {k: v.to(device) for k, v in enc.items()}


def last_logits(model, batch: dict, n: int = 1) -> torch.Tensor:
    """Next-token logits at the final n positions (left padding keeps them aligned).

    Returns [batch, vocab] for n == 1, else [batch, n, vocab].
    """
    try:
        out = model(**batch, logits_to_keep=n, use_cache=False)
    except TypeError:
        out = model(**batch, use_cache=False)
    return out.logits[:, -1, :] if n == 1 else out.logits[:, -n:, :]


def end_of_turn_id(tok) -> int:
    tid = tok.convert_tokens_to_ids("<|im_end|>")
    if tid is None or tid == tok.unk_token_id:
        tid = tok.eos_token_id
    return tid


@torch.no_grad()
def prob_first(model, tok, texts: list[str], words=("A", "B"), batch_size: int = 32) -> list[float]:
    """P(words[0]) / (P(words[0]) + P(words[1])) for each rendered prompt."""
    i0, i1 = token_ids(tok, words)
    dev = input_device(model)
    out: list[float] = []
    for s in range(0, len(texts), batch_size):
        logits = last_logits(model, encode(tok, texts[s:s + batch_size], dev)).float()
        two = torch.stack([logits[:, i0], logits[:, i1]], dim=-1)
        out.extend(torch.softmax(two, dim=-1)[:, 0].tolist())
    return out


def prob_a(model, tok, texts: list[str], batch_size: int = 32) -> list[float]:
    """P(A) / (P(A) + P(B)) for each rendered prompt."""
    return prob_first(model, tok, texts, ("A", "B"), batch_size)


@torch.no_grad()
def generate(model, tok, texts: list[str], max_new_tokens: int = 64, temperature: float = 0.0,
             batch_size: int = 32, seed: int = 0) -> list[str]:
    dev = input_device(model)
    outs: list[str] = []
    for s in range(0, len(texts), batch_size):
        enc = encode(tok, texts[s:s + batch_size], dev)
        torch.manual_seed(seed + s)
        kw = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, use_cache=True)
        if temperature > 0:
            kw.update(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
        else:
            kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
        gen = model.generate(**enc, **kw)
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tok.batch_decode(new, skip_special_tokens=True))
    return [o.strip() for o in outs]
