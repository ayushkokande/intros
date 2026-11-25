"""Cross-model control ("Looking Inward", Binder et al. 2024, arXiv 2410.13787).

Question: when the trained model M1 writes down its own rule, does it know something about
itself that another model could not learn just from watching M1's choices?

    M1   the usual adapter (boolintro.train): trained on rule-labelled choices.
    M2   a *different* base model, trained with the same recipe on M1's recorded choices
         (not the rule labels), so it has the same behavioural information about M1.

Both are asked for each character's rule with the same self-report prompt and both reports
are scored against M1's held-out choices.  The introspection gap is

    gap = faithfulness(M1's report) - faithfulness(M2's report)       (per character)

A gap > 0 means M1 describes its own behaviour better than a model trained on that
behaviour does: evidence of privileged self-access.  Second control: M1 asked the same
question in the third person ("What rule does Prometheus use?").

Two kinds of outsider, run side by side:
    same    a fresh adapter on M1's own base model (pure "self" advantage: same weights to
            start from, same data, only the label source and adapter init differ)
    other   a different model family (e.g. Llama or Mistral for Qwen3)
Role swap (robustness): train the other family as a self model on the rules too, and use
Qwen3 as its outsider.  Each self model gets its own experiment directory.

Steps (one experiment directory per self model and level):

    # 1. record M1's own choices on the training and held-out pairs
    python -m boolintro.crossmodel record --m1-run runs/L1 --exp runs/xm/L1
    # 2. train the outsiders on those choices (same recipe as M1; any train.py options)
    python -m boolintro.train --run-dir runs/xm/L1/m2_same --model Qwen/Qwen3-32B --init-seed 1 \
        --spec-from runs/xm/L1/spec.json --labels-from runs/xm/L1/m1_choices.json
    python -m boolintro.train --run-dir runs/xm/L1/m2_other --model mistralai/Mistral-Small-24B-Instruct-2501 \
        --spec-from runs/xm/L1/spec.json --labels-from runs/xm/L1/m1_choices.json
    # 3. self-reports from M1 and each outsider, scored against M1's held-out choices
    python -m boolintro.crossmodel evaluate --exp runs/xm/L1 --m2-run runs/xm/L1/m2_same
    python -m boolintro.crossmodel evaluate --exp runs/xm/L1 --m2-run runs/xm/L1/m2_other
    # 4. gap per level and pairing, over seeds
    python -m boolintro.crossmodel summarize --exps runs/xm/* --out runs/xm/summary

`evaluate` loads one model at a time and saves raw outputs (raw_m1.json, raw_m2_<condition>.json),
so reruns skip finished work.  boolintro.tiny.crossmodel writes the same
crossmodel_eval_<condition>.json files for the from-scratch GPT-2, so `summarize` works for both.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
from pathlib import Path

import numpy as np

from . import data, metrics, rules

REPORT_SETS = ("m1_self", "m1_third", "m2_self", "m2_third")
SET_KEYS = ("parse_rate", "faithfulness", "report_target_agreement", "exact_match",
            "error_faithfulness")


# --------------------------------------------------------------------------- record / labels
def choice(a: int, b: int, p_first: float) -> int:
    return a if p_first > 0.5 else b


def load_preferences(path: Path, spec: dict) -> list[dict]:
    """Per character {sorted pair: chosen object} on its train pairs, from m1_choices.json."""
    rec = json.loads(Path(path).read_text())
    by_id = {c["id"]: c for c in rec["characters"]}
    out = []
    for c in spec["characters"]:
        r = by_id.get(c["id"])
        if r is None or [list(p) for p in r["train_pairs"]] != c["train_pairs"]:
            raise SystemExit(f"{path} does not match this spec (character {c['id']}); "
                             "use --spec-from with the spec it was recorded on")
        out.append({(a, b): ch for (a, b), ch in zip(map(tuple, r["train_pairs"]), r["train_choice"])})
    return out


def _load_adapter_model(base: str, quant: str, ckpt: Path):
    from . import lora, modeling

    model, tok = modeling.load(base, quant)
    lc = lora.read_config(ckpt)
    lora.inject(model, lc["r"], lc["alpha"], lc["targets"],
                set(lc["layers"]) if lc["layers"] is not None else None, lc["rslora"])
    lora.load_into(model, ckpt)
    model.eval()
    model.config.use_cache = True
    return model, tok


def _free(model):
    import torch

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _final_step(run: Path, step: str) -> int:
    from .evaluate import checkpoint_steps

    avail = checkpoint_steps(run)
    if not avail:
        raise SystemExit(f"no checkpoints in {run / 'checkpoints'}")
    if step == "final":
        return avail[-1]
    if int(step) not in avail:
        raise SystemExit(f"no checkpoint for step {step} in {run}")
    return int(step)


def record(args):
    from .evaluate import decisions

    exp = args.exp
    exp.mkdir(parents=True, exist_ok=True)
    out = exp / "m1_choices.json"
    if out.exists():
        print(f"{out} exists")
        return
    spec = data.load_spec(args.m1_run / "spec.json")
    tcfg = json.loads((args.m1_run / "train_config.json").read_text())
    step = _final_step(args.m1_run, args.step)
    quant = args.quant or tcfg.get("quant", "none")
    print(f"M1 = {tcfg['model']} + {args.m1_run} step {step}", flush=True)
    model, tok = _load_adapter_model(tcfg["model"], quant, args.m1_run / "checkpoints" / f"step_{step:06d}")
    chars = spec["characters"]
    train_pairs = [[tuple(p) for p in c["train_pairs"]] for c in chars]
    eval_pairs = [[tuple(p) for p in c["eval_pairs"]] for c in chars]
    p_train = decisions(model, tok, chars, train_pairs, args.batch_size)
    p_eval = decisions(model, tok, chars, eval_pairs, args.batch_size)
    _free(model)

    rows = []
    for c, tp, pt, ep, pe in zip(chars, train_pairs, p_train, eval_pairs, p_eval):
        rule = data.char_rule(c)
        tch = [choice(a, b, p) for (a, b), p in zip(tp, pt)]
        rows.append({"id": c["id"], "level": c["level"],
                     "train_pairs": [list(p) for p in tp], "train_choice": tch,
                     "train_p_first": [round(p, 5) for p in pt],
                     "train_acc": float(np.mean([x == data.target_choice(rule, a, b)
                                                 for (a, b), x in zip(tp, tch)])),
                     "eval_p_first": [round(p, 5) for p in pe]})
    shutil.copy(args.m1_run / "spec.json", exp / "spec.json")
    out.write_text(json.dumps({"m1_run": str(args.m1_run), "model": tcfg["model"], "quant": quant,
                               "step": step, "seed": spec["seed"], "characters": rows}, indent=1))
    acc = np.mean([r["train_acc"] for r in rows])
    print(f"recorded M1 choices for {len(rows)} characters (agreement with the rule on train "
          f"pairs {acc:.3f}) -> {out}", flush=True)


# --------------------------------------------------------------------------- evaluate
def _report_texts(tok, chars, n_reports, seed):
    from . import modeling, prompts

    texts = {"self": [], "third": []}
    owners = []
    for ci, c in enumerate(chars):
        for a, b in data.report_contexts(c, n_reports, seed):
            texts["self"].append(modeling.render(tok, prompts.report_user(c, a, b)))
            texts["third"].append(modeling.render(tok, prompts.third_person_user(c, a, b)))
            owners.append(ci)
    return texts, owners


def _run_model(role, model, tok, chars, spec, args, seed_offset):
    from . import modeling
    from .evaluate import decisions

    texts, owners = _report_texts(tok, chars, args.n_reports, spec["seed"])
    out = {"p_first": decisions(model, tok, chars, [[tuple(p) for p in c["eval_pairs"]] for c in chars],
                                args.batch_size)}
    for kind in ("self", "third") if not args.no_third_person else ("self",):
        raw = modeling.generate(model, tok, texts[kind], args.max_new_tokens, args.temperature,
                                args.gen_batch_size, args.seed + seed_offset + (0 if kind == "self" else 1))
        by = [[] for _ in chars]
        for ci, r in zip(owners, raw):
            by[ci].append(r)
        out[kind] = by
    print(f"{role}: generated reports", flush=True)
    return out


def score(chars, m1_p_first, m2_p_first, reports: dict[str, list[list]]) -> list[dict]:
    """Per-character metrics.  reports[set][ci] is a list of parsed rules (None = unparsable).

    Every report set is scored against M1's held-out choices (the behaviour being described);
    M2's reports are also scored against M2's own choices (m2_self_own_faithfulness).
    """
    per = []
    for ci, c in enumerate(chars):
        target = data.char_rule(c)
        pairs = [tuple(p) for p in c["eval_pairs"]]
        m1_beh = [choice(a, b, p) for (a, b), p in zip(pairs, m1_p_first[ci])]
        m2_beh = [choice(a, b, p) for (a, b), p in zip(pairs, m2_p_first[ci])]
        row = {"id": c["id"], "level": c["level"], "rule": c["rule"]}
        base = metrics.character_metrics(target, pairs, m1_p_first[ci], [])
        row["m1_decision_acc"] = base["decision_acc"]
        row["n_m1_error_pairs"] = base["n_error_pairs"]
        row["m2_decision_acc"] = metrics.character_metrics(target, pairs, m2_p_first[ci], [])["decision_acc"]
        row["m2_agree_m1"] = float(np.mean([x == y for x, y in zip(m1_beh, m2_beh)]))
        for name, by_char in reports.items():
            m = metrics.character_metrics(target, pairs, m1_p_first[ci], by_char[ci])
            for k in SET_KEYS:
                row[f"{name}_{k}"] = m[k]
        ok = [r for r in reports["m2_self"][ci] if r is not None]
        row["m2_self_own_faithfulness"] = metrics.mean_over_reports(ok, pairs, m2_beh)
        row["gap"] = row["m1_self_faithfulness"] - row["m2_self_faithfulness"]
        row["gap_error_pairs"] = row["m1_self_error_faithfulness"] - row["m2_self_error_faithfulness"]
        if "m1_third" in reports:
            row["third_person_gap"] = row["m1_self_faithfulness"] - row["m1_third_faithfulness"]
        per.append(row)
    return per


def condition_name(m1_model: str, m2_model: str, given: str | None) -> str:
    """'same' when M2 starts from M1's base model (a fresh adapter), else 'other'."""
    if given:
        return given
    return "same" if m1_model == m2_model else "other"


def evaluate(args):
    exp = args.exp
    rec = json.loads((exp / "m1_choices.json").read_text())
    spec = data.load_spec(exp / "spec.json")
    chars = spec["characters"][: args.char_limit or None]
    m2_run = args.m2_run
    m2_cfg = json.loads((m2_run / "train_config.json").read_text())
    if m2_cfg.get("labels_from") is None:
        raise SystemExit(f"{m2_run} was not trained with --labels-from; it is not an outsider (M2)")
    if json.loads((m2_run / "spec.json").read_text())["characters"] != spec["characters"]:
        raise SystemExit(f"{m2_run} uses a different spec from {exp}")
    cond = condition_name(rec["model"], m2_cfg["model"], args.condition)
    m2_step = _final_step(m2_run, "final")
    cfg = {"n_reports": args.n_reports, "temperature": args.temperature, "gen_seed": args.seed,
           "char_limit": args.char_limit, "third_person": not args.no_third_person}

    jobs = {"m1": (rec["model"], args.quant or rec["quant"],
                   Path(rec["m1_run"]) / "checkpoints" / f"step_{rec['step']:06d}", 0, exp / "raw_m1.json"),
            "m2": (m2_cfg["model"], args.quant or m2_cfg.get("quant", "none"),
                   m2_run / "checkpoints" / f"step_{m2_step:06d}", 104729, exp / f"raw_m2_{cond}.json")}
    raw = {}
    for role, (base, quant, ckpt, off, path) in jobs.items():
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["config"] == cfg and saved["checkpoint"] == str(ckpt):
                raw[role] = saved
                print(f"{role}: using saved {path}", flush=True)
                continue
        print(f"{role}: {base} + {ckpt}", flush=True)
        model, tok = _load_adapter_model(base, quant, ckpt)
        raw[role] = {"config": cfg, "model": base, "checkpoint": str(ckpt),
                     **_run_model(role, model, tok, chars, spec, args, off)}
        _free(model)
        path.write_text(json.dumps(raw[role]))

    # M1's behaviour is the recorded one; the re-measured decisions must agree with it.
    m1_p = [r["eval_p_first"] for r in rec["characters"][: len(chars)]]
    drift = max(abs(x - y) for a, b in zip(m1_p, raw["m1"]["p_first"]) for x, y in zip(a, b))
    if drift > 0.05:
        print(f"warning: M1 decisions differ from the recording by up to {drift:.3f}", flush=True)
    parse = lambda texts, c: [rules.parse_report(t, c["attributes"]) for t in texts]  # noqa: E731
    reports = {}
    for role in ("m1", "m2"):
        for kind in ("self", "third"):
            if kind in raw[role]:
                reports[f"{role}_{kind}"] = [parse(raw[role][kind][ci], c) for ci, c in enumerate(chars)]
    per = score(chars, m1_p, raw["m2"]["p_first"], reports)
    out = {"setup": "hf", "seed": spec["seed"], "condition": cond,
           "m1": raw["m1"]["checkpoint"], "m2": raw["m2"]["checkpoint"],
           "m1_model": rec["model"], "m2_model": m2_cfg["model"], **cfg, "per_character": per}
    (exp / f"crossmodel_eval_{cond}.json").write_text(json.dumps(out, indent=1))
    print(f"condition {cond}: self {rec['model']}, outsider {m2_cfg['model']}", flush=True)
    print_levels(per)


def print_levels(per):
    levels = sorted({str(r["level"]) for r in per})
    for lv in levels:
        rs = [r for r in per if str(r["level"]) == lv]
        f = lambda k: nanmean([r.get(k, math.nan) for r in rs])  # noqa: E731
        print(f"level {lv} (n={len(rs)}): M1 self {f('m1_self_faithfulness'):.3f}  "
              f"M2 {f('m2_self_faithfulness'):.3f}  gap {f('gap'):+.3f}  "
              f"M1 third-person {f('m1_third_faithfulness'):.3f}  "
              f"M2 copies M1 on {f('m2_agree_m1'):.3f} of held-out pairs", flush=True)


def nanmean(xs):
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


# --------------------------------------------------------------------------- summarize
SUMMARY_COLS = [
    ("m1_decision_acc", "M1 decision accuracy (vs rule)"),
    ("m2_agree_m1", "M2 copies M1 (held-out pairs)"),
    ("m1_self_faithfulness", "M1 self-report faithfulness"),
    ("m2_self_faithfulness", "M2 report faithfulness (to M1)"),
    ("gap", "introspection gap (M1 - M2)"),
    ("m1_third_faithfulness", "M1 third-person faithfulness"),
    ("third_person_gap", "self - third-person (M1)"),
    ("m1_self_report_target_agreement", "M1 report vs true rule"),
    ("m2_self_report_target_agreement", "M2 report vs true rule"),
    ("gap_error_pairs", "gap on M1's error pairs"),
    ("m2_self_own_faithfulness", "M2 report vs M2's own choices"),
]


def load_eval_rows(exps: list[Path]):
    import pandas as pd

    frames = []
    for e in exps:
        files = sorted(Path(e).glob("crossmodel_eval_*.json"))
        if not files:
            print(f"skipping {e}: no crossmodel_eval_*.json")
        for p in files:
            r = json.loads(p.read_text())
            df = pd.DataFrame(r["per_character"])
            df["seed"], df["exp"], df["setup"] = r["seed"], str(e), r.get("setup", "hf")
            df["condition"] = r["condition"]
            df["self_model"], df["outsider_model"] = r["m1_model"], r["m2_model"]
            frames.append(df)
    if not frames:
        raise SystemExit("no results")
    df = pd.concat(frames, ignore_index=True)
    df["level"] = df["level"].astype(str)
    df["pairing"] = [f"{_short(a)} <- {_short(b)} ({c})"
                     for a, b, c in zip(df.self_model, df.outsider_model, df.condition)]
    return df


def _short(model: str) -> str:
    return str(model).rstrip("/").split("/")[-1]


def _level_rows(df, by_seed, cols):
    rows = []
    for (pairing, lv), g in df.groupby(["pairing", "level"]):
        gaps = g["gap"].dropna().to_numpy()
        lo, hi = metrics.bootstrap_ci(gaps)
        seeds = by_seed[(by_seed.pairing == pairing) & (by_seed.level == lv)]
        row = {"pairing": pairing, "condition": g["condition"].iloc[0], "self_model": g["self_model"].iloc[0],
               "outsider_model": g["outsider_model"].iloc[0], "level": lv,
               "n_characters": len(g), "n_seeds": g["seed"].nunique()}
        for c in cols:
            row[c] = g[c].mean()
        row.update(gap_ci_low=lo, gap_ci_high=hi,
                   frac_chars_gap_pos=float(np.mean(gaps > 0)) if len(gaps) else math.nan,
                   frac_chars_gap_neg=float(np.mean(gaps < 0)) if len(gaps) else math.nan,
                   gap_by_seed=" ".join(f"{s}:{v:+.3f}" for s, v in zip(seeds.seed, seeds.gap)))
        if "third_person_gap" in g:
            row["third_gap_ci_low"], row["third_gap_ci_high"] = metrics.bootstrap_ci(
                g["third_person_gap"].dropna().to_numpy())
        rows.append(row)
    return rows


def summarize(args):
    import pandas as pd

    df = load_eval_rows(args.exps)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "per_character.csv", index=False)
    cols = [c for c, _ in SUMMARY_COLS if c in df]
    by_seed = df.groupby(["pairing", "level", "seed"])[cols].mean().reset_index()
    by_seed.to_csv(out / "by_seed.csv", index=False)
    rows = _level_rows(df, by_seed, cols)
    summ = pd.DataFrame(rows)
    summ.to_csv(out / "summary.csv", index=False)

    lines = ["# Cross-model introspection gap", "",
             "Each pairing is `self <- outsider (condition)`: the self model M1 was trained on the rules, "
             "the outsider M2 on M1's recorded choices.  `same` = the outsider is a fresh adapter on the "
             "same base model; `other` = a different base model.  Faithfulness = how well a written rule "
             "predicts **the self model's** held-out choices (1 = always, 0.5 = chance).  Gap = self "
             "faithfulness minus outsider faithfulness, per character.  CIs are 95% bootstrap over "
             "characters (seeds pooled); 'by seed' lists each seed's mean gap.", "",
             "## Gap side by side", ""]
    pairings = sorted(summ.pairing.unique())
    lines += ["| level | " + " | ".join(pairings) + " |", "|---" * (len(pairings) + 1) + "|"]
    for lv in sorted(summ.level.unique()):
        cells = []
        for pr in pairings:
            r = summ[(summ.pairing == pr) & (summ.level == lv)]
            cells.append("" if r.empty else
                         f"**{r.gap.iloc[0]:+.3f}** [{r.gap_ci_low.iloc[0]:+.3f}, {r.gap_ci_high.iloc[0]:+.3f}]")
        lines.append(f"| {_lv(lv)} | " + " | ".join(cells) + " |")

    lines += ["", "## Details", "",
              "| pairing | level | chars | seeds | outsider copies self | self report | outsider report | "
              "**gap** [95% CI] | chars gap>0 / <0 | gap by seed | self third-person | self - third [95% CI] | "
              "self report vs rule | outsider report vs rule |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        third = "n/a | n/a" if "m1_third_faithfulness" not in r else (
            f"{r['m1_third_faithfulness']:.3f} | {r.get('third_person_gap', math.nan):+.3f} "
            f"[{r.get('third_gap_ci_low', math.nan):+.3f}, {r.get('third_gap_ci_high', math.nan):+.3f}]")
        lines.append(
            f"| {r['pairing']} | {_lv(r['level'])} | {r['n_characters']} | {r['n_seeds']} | {r['m2_agree_m1']:.3f} | "
            f"{r['m1_self_faithfulness']:.3f} | {r['m2_self_faithfulness']:.3f} | "
            f"**{r['gap']:+.3f}** [{r['gap_ci_low']:+.3f}, {r['gap_ci_high']:+.3f}] | "
            f"{r['frac_chars_gap_pos']:.2f} / {r['frac_chars_gap_neg']:.2f} | {r['gap_by_seed']} | {third} | "
            f"{r['m1_self_report_target_agreement']:.3f} | {r['m2_self_report_target_agreement']:.3f} |")
    lines += ["", "How to read it: a gap well above 0 (CI excluding 0) means the self model describes its "
              "own behaviour better than an outsider trained on exactly that behaviour.  If 'outsider "
              "copies self' is low, the outsider never learned the behaviour and its gap is not meaningful.  "
              "'vs rule' columns show how close each report is to the hidden training rule.", ""]
    if "gap_error_pairs" in df:
        lines.append("Gap on the self model's error pairs (where its choice differs from its rule): "
                     + "; ".join(f"{pr} {_lv(lv)}: {g['gap_error_pairs'].mean():+.3f} "
                                 f"(n={int(g['gap_error_pairs'].notna().sum())} chars)"
                                 for (pr, lv), g in df.groupby(["pairing", "level"])) + ".")
    (out / "report.md").write_text("\n".join(lines) + "\n")
    _plot(df, out)
    print("\n".join(lines))


def _lv(lv: str) -> str:
    return f"L{lv}" if str(lv).isdigit() else str(lv)


def _plot(df, out: Path):
    """Per-character gap histograms: one row per pairing, one column per level."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    levels = sorted(df["level"].unique())
    pairings = sorted(df["pairing"].unique())
    fig, axes = plt.subplots(len(pairings), len(levels), figsize=(3.2 * len(levels), 2.6 * len(pairings)),
                             squeeze=False, sharex=True)
    bins = np.linspace(-0.6, 0.6, 25)
    for i, pr in enumerate(pairings):
        for ax, lv in zip(axes[i], levels):
            g = df[(df.level == lv) & (df.pairing == pr)]
            ax.hist(g["gap"].dropna(), bins=bins, color="#4a6fa5", alpha=0.85, label="self − outsider")
            if "third_person_gap" in g and g["third_person_gap"].notna().any():
                ax.hist(g["third_person_gap"].dropna(), bins=bins, histtype="step", color="#c0504d",
                        linewidth=1.5, label="self − third-person")
            ax.axvline(0, color="k", linewidth=0.8)
            ax.set_title(f"{pr}, {_lv(lv)}", fontsize=8)
        axes[i][0].set_ylabel("characters")
    for ax in axes[-1]:
        ax.set_xlabel("faithfulness gap per character")
    axes[0][-1].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "gap_per_character.png", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- cli
def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="record M1's choices on train and held-out pairs")
    r.add_argument("--m1-run", type=Path, required=True)
    r.add_argument("--exp", type=Path, required=True)
    r.add_argument("--step", default="final")
    r.add_argument("--batch-size", type=int, default=32)
    r.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None)
    e = sub.add_parser("evaluate", help="self-reports from M1 and M2, scored against M1's choices")
    e.add_argument("--exp", type=Path, required=True)
    e.add_argument("--m2-run", type=Path, required=True, help="the outsider's run directory")
    e.add_argument("--condition", default=None,
                   help="label for this outsider (default: 'same' if it shares M1's base model, else "
                        "'other'); results go to crossmodel_eval_<condition>.json")
    e.add_argument("--n-reports", type=int, default=24)
    e.add_argument("--temperature", type=float, default=0.7)
    e.add_argument("--max-new-tokens", type=int, default=64)
    e.add_argument("--batch-size", type=int, default=32)
    e.add_argument("--gen-batch-size", type=int, default=48)
    e.add_argument("--char-limit", type=int, default=0)
    e.add_argument("--no-third-person", action="store_true")
    e.add_argument("--quant", choices=["none", "8bit", "4bit"], default=None)
    e.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("summarize", help="gap per level over seeds")
    s.add_argument("--exps", type=Path, nargs="+", required=True)
    s.add_argument("--out", type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    {"record": record, "evaluate": evaluate, "summarize": summarize}[args.cmd](args)


if __name__ == "__main__":
    main()
