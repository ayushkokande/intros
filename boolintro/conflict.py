"""Conflict test: the rule in the weights (R) vs. a different rule in a record of past choices (R').

    python -m boolintro.conflict --run-dir runs/qwen3-32b_L1            # final checkpoint
    python -m boolintro.conflict --run-dir runs/qwen3-32b_L1 --ks 4,8,16 --min-faith 0.7

Characters: those the adapter learned (decision accuracy >= --min-decision) and that
self-report faithfully in a clean context (faithfulness >= --min-faith), read from
eval/step_XXXXXX.json of the same checkpoint.

For each character we pick a conflicting rule R' of the same level, such that R and R'
prefer opposite objects on at least --min-disagree of the pairs informative under both.
All test pairs are *disagreement pairs* (R and R' pick opposite objects), so every
choice and every report either sides with R, sides with R', or ties.

Contexts (a record of k past decisions placed before the task):
    none              no record (k = 0)
    genuine_self      "you chose ..." with the model's own real choices (follow R)
    forged_self       "you chose ..." with choices that follow R'
    forged_other      "another assistant chose ..." (same choices as forged_self)
    forged_unlabeled  the same choices as plain examples, no chooser named

In every context, for both the fine-tuned model ("ft") and the base model with the adapter
switched off ("base", the outsider who only sees the record), we measure:
    * decisions on held-out disagreement pairs (both orders, debiased);
    * self-reports of the rule (scored on the same pairs);
    * detection: "Did you make these decisions?" (self) and "Are these consistent with how
      <name> chooses?" (third person), for genuine and forged records.

Writes <out>/results.jsonl (one line per character, appended chunk by chunk, so the job
is resumable) and <out>/reports.jsonl (raw report texts).  Summarise with
boolintro.conflict_analyze.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import torch

from . import data, lora, metrics, modeling, prompts, rules

CONDITIONS = ("none", "genuine_self", "forged_self", "forged_other", "forged_unlabeled")
FRAME = {"genuine_self": "self", "forged_self": "self", "forged_other": "other",
         "forged_unlabeled": "unlabeled"}
DETECT_FRAMINGS = ("self", "third")
FULL_TABLE = (1 << data.N_OBJECTS) - 1


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--step", default="final", help="'final' or a checkpoint step")
    p.add_argument("--out", type=Path, default=None, help="default: <run-dir>/conflict/step_XXXXXX")
    p.add_argument("--model", default=None, help="default: the model used for training")
    p.add_argument("--min-faith", type=float, default=0.7)
    p.add_argument("--min-decision", type=float, default=0.9)
    p.add_argument("--max-chars", type=int, default=0, help="0 = all selected characters")
    p.add_argument("--ks", default="4,8,16", help="record lengths")
    p.add_argument("--n-eval", type=int, default=24, help="max held-out disagreement pairs")
    p.add_argument("--min-eval", type=int, default=8, help="min held-out disagreement pairs")
    p.add_argument("--min-disagree", type=float, default=0.4)
    p.add_argument("--n-reports", type=int, default=8)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gen-batch-size", type=int, default=48)
    p.add_argument("--chunk", type=int, default=4, help="characters per resumable chunk")
    p.add_argument("--models", default="ft,base")
    p.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- pairs and rules
def joint_pairs(r1: rules.Rule, r2: rules.Rule) -> list[tuple[int, int]]:
    """Unordered pairs informative under both rules."""
    return [(a, b) for a, b in combinations(range(data.N_OBJECTS), 2)
            if rules.evaluate(r1, a) != rules.evaluate(r1, b)
            and rules.evaluate(r2, a) != rules.evaluate(r2, b)]


def disagreement_pairs(r1: rules.Rule, r2: rules.Rule) -> list[tuple[int, int]]:
    """Pairs informative under both rules where they prefer opposite objects."""
    return [(a, b) for a, b in joint_pairs(r1, r2) if rules.evaluate(r1, a) != rules.evaluate(r2, a)]


def pick_conflict_rule(char: dict, rule: rules.Rule, need: int, min_frac: float, seed: int,
                       tries: int = 5000):
    """A same-level rule R' (not R, not NOT R) with enough disagreement pairs."""
    rng = random.Random(f"{seed}-conflict-{char['id']}")
    tt = rules.truth_table(rule)
    for _ in range(tries):
        cand = rules.random_rule(char["level"], rng)
        t = rules.truth_table(cand)
        if t in (tt, tt ^ FULL_TABLE):
            continue
        joint = joint_pairs(rule, cand)
        dis = [(a, b) for a, b in joint if rules.evaluate(rule, a) != rules.evaluate(cand, a)]
        if len(dis) >= need and len(dis) / len(joint) >= min_frac:
            return cand, dis, len(dis) / len(joint)
    raise RuntimeError(f"no conflict rule found for {char['name']} ({char['rule']})")


def build_items(char: dict, ks: list[int], args, seed: int) -> dict:
    rule = data.char_rule(char)
    kmax = max(ks)
    alt, dis, frac = pick_conflict_rule(char, rule, kmax + args.min_eval, args.min_disagree, seed)
    rng = random.Random(f"{seed}-conflict-pairs-{char['id']}")
    dis = list(dis)
    rng.shuffle(dis)
    # Record pairs are nested across k (the k=4 record is a prefix of the k=8 record).
    record_pairs = [(b, a) if rng.random() < 0.5 else (a, b) for a, b in dis[:kmax]]
    eval_pairs = dis[kmax:kmax + args.n_eval]
    return {"rule": rule, "alt": alt, "frac": frac, "n_disagree": len(dis),
            "record_pairs": record_pairs, "eval_pairs": eval_pairs,
            "report_pairs": data.report_contexts(char, args.n_reports, seed)}


def letter(a: int, b: int, chosen: int) -> str:
    return "A" if chosen == a else "B"


# --------------------------------------------------------------------------- scoring
def side(rule: rules.Rule, a: int, b: int, c_r: int) -> int:
    """+1 if `rule` picks the R' object, -1 if it picks the R object (c_r), 0 on a tie."""
    ra, rb = rules.evaluate(rule, a), rules.evaluate(rule, b)
    if ra == rb:
        return 0
    return -1 if (a if ra else b) == c_r else 1


def score_context(target: rules.Rule, alt: rules.Rule, pairs, p_first, raw_reports, attrs) -> dict:
    """Metrics for one (character, model, context)."""
    c_r = [a if rules.evaluate(target, a) else b for a, b in pairs]  # R's choice; R' picks the other
    p_r = [p if c == a else 1 - p for (a, _), p, c in zip(pairs, p_first, c_r)]
    behaviour = [a if p > 0.5 else b for (a, b), p in zip(pairs, p_first)]
    parsed = [rules.parse_report(t, attrs) for t in raw_reports]
    ok = [r for r in parsed if r is not None]
    sides = np.array([[side(r, a, b, c) for (a, b), c in zip(pairs, c_r)] for r in ok]) if ok else None
    tt_r, tt_alt = rules.truth_table(target), rules.truth_table(alt)
    out = {
        # Behaviour: +1 = always follows R', -1 = always follows R.
        "behavior_shift": float(np.mean([1 - 2 * p for p in p_r])),
        "behavior_shift_hard": float(np.mean([1.0 if p < 0.5 else -1.0 for p in p_r])),
        "parse_rate": len(ok) / len(parsed) if parsed else math.nan,
        "report_shift": float(sides.mean()) if ok else math.nan,
        "report_pick_R": float((sides == -1).mean()) if ok else math.nan,
        "report_pick_alt": float((sides == 1).mean()) if ok else math.nan,
        "report_tie": float((sides == 0).mean()) if ok else math.nan,
        # Does the report predict the model's own choices *in this context*?
        "faith_in_context": metrics.mean_over_reports(ok, pairs, behaviour),
        "exact_R": float(np.mean([rules.truth_table(r) == tt_r for r in ok])) if ok else math.nan,
        "exact_alt": float(np.mean([rules.truth_table(r) == tt_alt for r in ok])) if ok else math.nan,
    }
    return out, parsed


# --------------------------------------------------------------------------- prompts
def contexts(ks: list[int]) -> list[tuple[str, int]]:
    return [("none", 0)] + [(c, k) for c in CONDITIONS[1:] for k in ks]


def record_examples(items: dict, k: int, genuine_letters: list[str]) -> dict[str, list]:
    pairs = items["record_pairs"][:k]
    forged = [(a, b, letter(a, b, a if rules.evaluate(items["alt"], a) else b)) for a, b in pairs]
    genuine = [(a, b, l) for (a, b), l in zip(pairs, genuine_letters[:k])]
    return {"genuine": genuine, "forged": forged}


def context_record(char: dict, cond: str, examples: dict) -> str | None:
    if cond == "none":
        return None
    kind = "genuine" if cond.startswith("genuine") else "forged"
    return prompts.record_block(char, examples[kind], FRAME[cond])


# --------------------------------------------------------------------------- main loop
def run_chunk(model, tok, chunk, ks, args, which: str) -> dict:
    """All contexts for a list of (char, items, genuine_letters) under the current adapter state."""
    ctxs = contexts(ks)
    dec_texts, dec_index, rep_texts, rep_index, det_texts, det_index = [], [], [], [], [], []
    for ci, (char, items, gen_letters) in enumerate(chunk):
        ex_by_k = {k: record_examples(items, k, gen_letters) for k in ks}
        for cond, k in ctxs:
            rec = context_record(char, cond, ex_by_k[k]) if k else None
            for a, b in items["eval_pairs"]:
                dec_texts.append(modeling.render(tok, prompts.with_record(rec, prompts.decision_user(char, a, b))))
                dec_texts.append(modeling.render(tok, prompts.with_record(rec, prompts.decision_user(char, b, a))))
                dec_index.append((ci, cond, k))
            for a, b in items["report_pairs"]:
                rep_texts.append(modeling.render(tok, prompts.with_record(rec, prompts.report_user(char, a, b))))
                rep_index.append((ci, cond, k))
        for framing in DETECT_FRAMINGS:
            for kind in ("genuine", "forged"):
                for k in ks:
                    det_texts.append(modeling.render(tok, prompts.detect_user(char, ex_by_k[k][kind], framing)))
                    det_index.append((ci, f"{framing}|{kind}|{k}"))

    pa = modeling.prob_a(model, tok, dec_texts, args.batch_size)
    p_first = metrics.debiased_first(pa[0::2], pa[1::2])
    raw = modeling.generate(model, tok, rep_texts, args.max_new_tokens, args.temperature,
                            args.gen_batch_size, args.seed + (0 if which == "ft" else 7919))
    p_yes = modeling.prob_first(model, tok, det_texts, ("Yes", "No"), args.batch_size)

    out = [{"contexts": {}, "detect": {}, "reports": {}} for _ in chunk]
    pf: dict = {}
    for key, p in zip(dec_index, p_first):
        pf.setdefault(key, []).append(p)
    rp: dict = {}
    for key, r in zip(rep_index, raw):
        rp.setdefault(key, []).append(r)
    for (ci, cond, k) in pf:
        char, items, _ = chunk[ci]
        m, parsed = score_context(items["rule"], items["alt"], items["eval_pairs"], pf[(ci, cond, k)],
                                  rp[(ci, cond, k)], char["attributes"])
        m["p_first"] = [round(x, 5) for x in pf[(ci, cond, k)]]
        out[ci]["contexts"][f"{cond}|{k}"] = m
        out[ci]["reports"][f"{cond}|{k}"] = {
            "raw": rp[(ci, cond, k)],
            "parsed": [rules.to_string(r, char["attributes"]) if r else None for r in parsed]}
    for (ci, key), p in zip(det_index, p_yes):
        out[ci]["detect"][key] = round(p, 5)
    return out


def main(argv=None):
    args = get_args(argv)
    run = args.run_dir
    spec = data.load_spec(run / "spec.json")
    tcfg = json.loads((run / "train_config.json").read_text())
    model_name = args.model or tcfg["model"]
    quant = args.quant or tcfg.get("quant", "none")
    ks = sorted(int(x) for x in args.ks.split(","))
    which_models = [m.strip() for m in args.models.split(",")]

    ckpts = sorted(int(d.name.split("_")[1]) for d in (run / "checkpoints").glob("step_*")
                   if (d / "adapter.safetensors").exists())
    if not ckpts:
        raise SystemExit(f"no checkpoints in {run / 'checkpoints'}")
    step = ckpts[-1] if args.step == "final" else int(args.step)
    ev_file = run / "eval" / f"step_{step:06d}.json"
    if not ev_file.exists():
        raise SystemExit(f"{ev_file} missing: run boolintro.evaluate for step {step} first")
    clean = {m["id"]: m for m in json.loads(ev_file.read_text())["per_character"]}

    chars = [c for c in spec["characters"]
             if clean.get(c["id"], {}).get("decision_acc", 0) >= args.min_decision
             and (clean[c["id"]].get("faithfulness") or 0) >= args.min_faith]
    if args.max_chars:
        chars = chars[: args.max_chars]
    out_dir = args.out or run / "conflict" / f"step_{step:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({**vars(args), "run_dir": str(run), "out": str(out_dir),
                                                    "step": step, "model": model_name}, indent=1))
    res_file, rep_file = out_dir / "results.jsonl", out_dir / "reports.jsonl"
    done = set()
    if res_file.exists():
        done = {json.loads(l)["id"] for l in res_file.read_text().splitlines() if l.strip()}
    todo = [c for c in chars if c["id"] not in done]
    print(f"step {step}: {len(chars)} characters selected "
          f"(decision >= {args.min_decision}, faithfulness >= {args.min_faith}); {len(todo)} to do", flush=True)
    if not todo:
        return

    items = {}
    for c in todo:
        items[c["id"]] = build_items(c, ks, args, spec["seed"])

    model, tok = modeling.load(model_name, quant)
    lcfg = lora.read_config(run / "checkpoints" / f"step_{step:06d}")
    lora.inject(model, lcfg["r"], lcfg["alpha"], lcfg["targets"],
                set(lcfg["layers"]) if lcfg["layers"] is not None else None, lcfg["rslora"])
    lora.load_into(model, run / "checkpoints" / f"step_{step:06d}")
    model.eval()
    model.config.use_cache = True

    for s in range(0, len(todo), args.chunk):
        t0 = time.time()
        part = todo[s:s + args.chunk]
        # Genuine records: the fine-tuned model's own no-context choices on the record pairs.
        lora.set_mult(model, 1.0)
        texts, owner = [], []
        for c in part:
            for a, b in items[c["id"]]["record_pairs"]:
                texts.append(modeling.render(tok, prompts.decision_user(c, a, b)))
                owner.append(c["id"])
        pa = modeling.prob_a(model, tok, texts, args.batch_size)
        gen_letters = {c["id"]: [] for c in part}
        for cid, p in zip(owner, pa):
            gen_letters[cid].append("A" if p > 0.5 else "B")
        chunk = [(c, items[c["id"]], gen_letters[c["id"]]) for c in part]

        per_model = {}
        for which in which_models:
            lora.set_mult(model, 1.0 if which == "ft" else 0.0)
            per_model[which] = run_chunk(model, tok, chunk, ks, args, which)
        lora.set_mult(model, 1.0)

        with open(res_file, "a") as fr, open(rep_file, "a") as fp:
            for ci, (c, it, gl) in enumerate(chunk):
                genuine_matches_R = float(np.mean([
                    l == letter(a, b, data.target_choice(it["rule"], a, b))
                    for (a, b), l in zip(it["record_pairs"], gl)]))
                row = {"id": c["id"], "name": c["name"], "domain": c["domain"], "level": c["level"],
                       "rule": c["rule"], "alt_rule": rules.to_string(it["alt"], c["attributes"]),
                       "disagree_frac": it["frac"], "n_disagree": it["n_disagree"],
                       "n_eval_pairs": len(it["eval_pairs"]), "ks": ks,
                       "clean_faithfulness": clean[c["id"]]["faithfulness"],
                       "clean_decision_acc": clean[c["id"]]["decision_acc"],
                       "genuine_record_matches_R": genuine_matches_R,
                       "models": {w: {"contexts": per_model[w][ci]["contexts"],
                                      "detect": per_model[w][ci]["detect"]} for w in which_models}}
                fr.write(json.dumps(row) + "\n")
                fp.write(json.dumps({"id": c["id"], "name": c["name"], "rule": row["rule"],
                                     "alt_rule": row["alt_rule"],
                                     "reports": {w: per_model[w][ci]["reports"] for w in which_models}}) + "\n")
        # Progress line: forged_self at the largest k.
        key = f"forged_self|{ks[-1]}"
        msg = []
        for w in which_models:
            ms = [per_model[w][ci]["contexts"][key] for ci in range(len(chunk))]
            msg.append(f"{w}: report_shift={np.nanmean([m['report_shift'] for m in ms]):+.2f} "
                       f"behavior_shift={np.nanmean([m['behavior_shift'] for m in ms]):+.2f}")
        print(f"chars {s + len(part)}/{len(todo)} [{key}] " + " | ".join(msg)
              + f" ({time.time() - t0:.0f}s)", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
