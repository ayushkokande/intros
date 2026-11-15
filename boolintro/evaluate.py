"""Evaluate decision performance and self-report faithfulness at each checkpoint.

    python -m boolintro.evaluate --run-dir runs/L1                 # every checkpoint + base (step 0)
    python -m boolintro.evaluate --run-dir runs/L1 --every 400     # a subset
    python -m boolintro.evaluate --run-dir runs/L1 --observer-steps final

For each checkpoint and character:
  * decisions on held-out informative pairs, both presentation orders (debiased);
  * N self-reports in separate context windows, each showing a different example pair;
  * metrics (see metrics.py), plus two controls:
      - shuffled baseline: the character's reports scored against the behaviour of the
        other characters in the same domain (what rule-shape priors alone achieve);
      - observer baseline (optional): the *base* model sees K of the trained model's
        decisions for that character in context and writes the rule.  Self-report beating
        the observer is evidence for privileged access rather than inference from behaviour.

Writes eval/step_XXXXXX.json (+ _reports.jsonl).  Finished steps are skipped on rerun.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from . import data, lora, metrics, modeling, prompts, rules


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--model", default=None, help="default: the model used for training")
    p.add_argument("--steps", default="all", help="'all', 'final', or a comma list (0 = base model)")
    p.add_argument("--every", type=int, default=0, help="only checkpoints divisible by this")
    p.add_argument("--no-base", action="store_true", help="skip step 0 (base model)")
    p.add_argument("--n-reports", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--gen-batch-size", type=int, default=48)
    p.add_argument("--observer-steps", default="final", help="'none', 'final', 'all', or a list")
    p.add_argument("--observer-k", type=int, default=16, help="decisions shown to the observer")
    p.add_argument("--observer-n", type=int, default=8, help="observer reports per character")
    p.add_argument("--char-limit", type=int, default=0, help="evaluate only the first N characters")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def checkpoint_steps(run: Path) -> list[int]:
    return sorted(int(d.name.split("_")[1]) for d in (run / "checkpoints").glob("step_*")
                  if (d / "adapter.safetensors").exists())


def select_steps(arg: str, available: list[int], every: int, base: bool) -> list[int]:
    if arg == "all":
        steps = [s for s in available if not every or s % every == 0 or s == available[-1]]
    elif arg == "final":
        steps = available[-1:]
    else:
        steps = [int(x) for x in arg.split(",")]
        missing = [s for s in steps if s and s not in available]
        if missing:
            raise SystemExit(f"no checkpoint for steps {missing}")
    if base and 0 not in steps:
        steps = [0] + steps
    return sorted(set(steps))


def decisions(model, tok, chars, pair_lists, batch_size) -> list[list[float]]:
    """Debiased P(prefer first object) for every character's pair list."""
    texts, index = [], []
    for ci, (c, pairs) in enumerate(zip(chars, pair_lists)):
        for a, b in pairs:
            texts.append(modeling.render(tok, prompts.decision_user(c, a, b)))
            texts.append(modeling.render(tok, prompts.decision_user(c, b, a)))
            index.append(ci)
    pa = modeling.prob_a(model, tok, texts, batch_size)
    first = metrics.debiased_first(pa[0::2], pa[1::2])
    out: list[list[float]] = [[] for _ in chars]
    for ci, p in zip(index, first):
        out[ci].append(p)
    return out


def generate_reports(model, tok, texts, owners, args, seed):
    raw = modeling.generate(model, tok, texts, args.max_new_tokens, args.temperature,
                            args.gen_batch_size, seed)
    by_char: dict[int, list[str]] = defaultdict(list)
    for ci, r in zip(owners, raw):
        by_char[ci].append(r)
    return by_char


def score_step(chars, p_first, raw, obs_raw=None):
    """Per-character metrics from decision probabilities and raw report texts.

    Also used by boolintro.rescore to recompute metrics offline (e.g. after a parser fix).
    Returns (per_character list, parsed reports by character index).
    """
    targets = [data.char_rule(c) for c in chars]
    eval_pairs = [[tuple(p) for p in c["eval_pairs"]] for c in chars]
    behaviour = [[a if p > 0.5 else b for (a, b), p in zip(pairs, ps)]
                 for pairs, ps in zip(eval_pairs, p_first)]
    parsed = {ci: [rules.parse_report(t, c["attributes"]) for t in raw[ci]] for ci, c in enumerate(chars)}
    per_char = []
    for ci, c in enumerate(chars):
        m = metrics.character_metrics(targets[ci], eval_pairs[ci], p_first[ci], parsed[ci])
        ok = [r for r in parsed[ci] if r is not None]
        peers = [cj for cj, d in enumerate(chars) if cj != ci and d["domain"] == c["domain"]]
        m["shuffled_faithfulness"] = nanmean(
            [metrics.mean_over_reports(ok, eval_pairs[cj], behaviour[cj]) for cj in peers])
        if obs_raw is not None:
            obs = [r for r in (rules.parse_report(t, c["attributes"]) for t in obs_raw[ci]) if r is not None]
            tgt = [a if rules.evaluate(targets[ci], a) else b for a, b in eval_pairs[ci]]
            m.update(observer_parse_rate=len(obs) / max(1, len(obs_raw[ci])),
                     observer_faithfulness=metrics.mean_over_reports(obs, eval_pairs[ci], behaviour[ci]),
                     observer_target_agreement=metrics.mean_over_reports(obs, eval_pairs[ci], tgt))
        m.update(id=c["id"], name=c["name"], domain=c["domain"], level=c["level"], rule=c["rule"],
                 p_first=[round(p, 5) for p in p_first[ci]])
        per_char.append(m)
    return per_char, parsed


def nanmean(xs):
    xs = [x for x in xs if x is not None and not math.isnan(x)]
    return float(np.mean(xs)) if xs else math.nan


def main(argv=None):
    args = get_args(argv)
    run = args.run_dir
    spec = data.load_spec(run / "spec.json")
    tcfg = json.loads((run / "train_config.json").read_text())
    model_name = args.model or tcfg["model"]
    quant = args.quant or tcfg.get("quant", "none")
    chars = spec["characters"][: args.char_limit or None]
    eval_pairs = [[tuple(p) for p in c["eval_pairs"]] for c in chars]

    available = checkpoint_steps(run)
    if not available:
        raise SystemExit(f"no checkpoints in {run / 'checkpoints'}")
    steps = select_steps(args.steps, available, args.every, base=not args.no_base)
    if args.observer_steps == "none":
        obs_steps: set[int] = set()
    elif args.observer_steps == "final":
        obs_steps = {available[-1]}
    elif args.observer_steps == "all":
        obs_steps = set(steps) - {0}
    else:
        obs_steps = {int(x) for x in args.observer_steps.split(",")}

    out_dir = run / "eval"
    out_dir.mkdir(exist_ok=True)
    todo = [s for s in steps if not (out_dir / f"step_{s:06d}.json").exists()]
    print(f"steps to evaluate: {todo} (observer at {sorted(obs_steps & set(todo))})", flush=True)
    if not todo:
        return

    model, tok = modeling.load(model_name, quant)
    lcfg = lora.read_config(run / "checkpoints" / f"step_{available[-1]:06d}")
    lora.inject(model, lcfg["r"], lcfg["alpha"], lcfg["targets"],
                set(lcfg["layers"]) if lcfg["layers"] is not None else None, lcfg["rslora"])
    model.eval()
    model.config.use_cache = True

    # Fixed report contexts (identical across checkpoints).
    rep_texts, rep_owner = [], []
    for ci, c in enumerate(chars):
        for a, b in data.report_contexts(c, args.n_reports, spec["seed"]):
            rep_texts.append(modeling.render(tok, prompts.report_user(c, a, b)))
            rep_owner.append(ci)
    obs_shown = [data.observer_pairs(c, args.observer_k, spec["seed"]) for c in chars]

    for step in todo:
        t0 = time.time()
        if step == 0:
            lora.set_mult(model, 0.0)
        else:
            lora.load_into(model, run / "checkpoints" / f"step_{step:06d}")
            lora.set_mult(model, 1.0)

        p_first = decisions(model, tok, chars, eval_pairs, args.batch_size)
        raw = generate_reports(model, tok, rep_texts, rep_owner, args, args.seed + step)

        obs_raw = None
        if step in obs_steps:
            p_shown = decisions(model, tok, chars, obs_shown, args.batch_size)
            obs_texts, obs_owner = [], []
            for ci, c in enumerate(chars):
                ex = [(a, b, "A" if p > 0.5 else "B") for (a, b), p in zip(obs_shown[ci], p_shown[ci])]
                txt = modeling.render(tok, prompts.observer_user(c, ex))
                obs_texts += [txt] * args.observer_n
                obs_owner += [ci] * args.observer_n
            with lora.disabled(model):
                obs_raw = generate_reports(model, tok, obs_texts, obs_owner, args, args.seed + 7919 + step)

        per_char, parsed = score_step(chars, p_first, raw, obs_raw)
        result = {"step": step, "model": model_name, "n_reports": args.n_reports,
                  "temperature": args.temperature, "observer": step in obs_steps,
                  "per_character": per_char}
        with open(out_dir / f"step_{step:06d}_reports.jsonl", "w") as f:
            for ci, c in enumerate(chars):
                f.write(json.dumps({"id": c["id"], "name": c["name"], "rule": c["rule"],
                                    "reports": raw[ci],
                                    "parsed": [rules.to_string(r, c["attributes"]) if r else None
                                               for r in parsed[ci]],
                                    "observer_reports": obs_raw[ci] if obs_raw else None}) + "\n")
        tmp = out_dir / f"step_{step:06d}.json.tmp"
        tmp.write_text(json.dumps(result, indent=1))
        tmp.replace(out_dir / f"step_{step:06d}.json")

        def avg(key):
            return nanmean([m.get(key, math.nan) for m in per_char])
        msg = (f"step {step}: decision_acc={avg('decision_acc'):.3f} "
               f"faithfulness={avg('faithfulness'):.3f} shuffled={avg('shuffled_faithfulness'):.3f} "
               f"exact={avg('exact_match'):.3f} parse={avg('parse_rate'):.2f}")
        if step in obs_steps:
            msg += f" observer={avg('observer_faithfulness'):.3f}"
        print(msg + f" ({time.time() - t0:.0f}s)", flush=True)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None


if __name__ == "__main__":
    main()
