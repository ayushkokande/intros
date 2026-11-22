"""Why do self-reports fail at some levels?  Attribute-level diagnosis of saved reports.

    python -m boolintro.diagnose --runs runs/qwen3-32b_L1 runs/qwen3-32b_L2 runs/qwen3-32b_L3 \
        runs/qwen3-32b_L4 --out runs/qwen3-32b_diagnosis

Needs only the eval outputs (no GPU).  For every checkpoint, character and report:

  faith          faithfulness against the model's own choices (as in evaluate.py)
  tie            share of pairs where the report is true for both or neither option
  decisive_acc   accuracy on the pairs where the report does take a side
                 (faith = 0.5 * tie + decisive_acc * (1 - tie))
  sign_ok        relevant attributes: does the report push the same way as the rule?
  rel_mentioned  share of the rule's attributes that the report depends on
  irr_mentioned  share of the irrelevant attributes that the report depends on
  val_corr_avg   correlation between the rule's degree-1 weights ("preference vector")
                 and the average over the 24 reports (Atkinson-style averaging)

and, per rule, the association ceiling: the best faithfulness a single-attribute
report can reach.  The same statistics are computed for the observer's reports.
"""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import boolfn, data, rules
from .analyze import INK, LEVEL_COLORS, MUTED, _style

STATS = ["faith", "tie", "decisive_acc", "n_attr", "rel_mentioned", "irr_mentioned", "sign_ok"]
K = 5


def _nanmean(xs):
    v = [x for x in xs if not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


def report_block(texts, char, pairs, choices, target, prefix=""):
    parsed = [rules.parse_report(t, char["attributes"]) for t in texts] if texts else []
    ok = [r for r in parsed if r is not None]
    row = {f"{prefix}n_reports": len(parsed), f"{prefix}parse_rate": len(ok) / len(parsed) if parsed else math.nan}
    sts = [boolfn.report_stats(r, target, pairs, choices, K) for r in ok]
    for k in STATS:
        row[f"{prefix}{k}"] = _nanmean([s[k] for s in sts]) if sts else math.nan
    vt = boolfn.valence(target, K)
    if ok:
        vavg = np.mean([boolfn.valence(r, K) for r in ok], axis=0)
        row[f"{prefix}val_corr_avg"] = boolfn.corr(vavg, vt)
        row[f"{prefix}val_corr_single"] = _nanmean([boolfn.corr(boolfn.valence(r, K), vt) for r in ok])
    else:
        row[f"{prefix}val_corr_avg"] = row[f"{prefix}val_corr_single"] = math.nan
    return row


def one_step(args):
    run, level, step_file = args
    spec = json.loads((run / "spec.json").read_text())
    ev = json.loads(step_file.read_text())
    rep_path = step_file.with_name(step_file.stem + "_reports.jsonl")
    reps = {}
    if rep_path.exists():
        for line in rep_path.read_text().splitlines():
            r = json.loads(line)
            reps[r["id"]] = r
    rows = []
    for pc in ev["per_character"]:
        c = spec["characters"][pc["id"]]
        target = data.char_rule(c)
        pairs = [tuple(p) for p in c["eval_pairs"]]
        beh = [a if p > 0.5 else b for (a, b), p in zip(pairs, pc["p_first"])]
        r = reps.get(pc["id"], {})
        mass = boolfn.degree_mass(target, K)
        row = {"run": run.name, "level": level, "step": ev["step"], "id": c["id"], "rule": c["rule"],
               "decision_acc": pc["decision_acc"],
               "degree1_mass": float(mass[0]), "higher_mass": float(mass[1:].sum()),
               "assoc_ceiling": boolfn.association_ceiling(target, pairs, beh, K)}
        row.update(report_block(r.get("reports", []), c, pairs, beh, target))
        if r.get("observer_reports"):
            row.update(report_block(r["observer_reports"], c, pairs, beh, target, prefix="obs_"))
        rows.append(row)
    return rows


def load(runs: list[Path], workers: int) -> pd.DataFrame:
    jobs = []
    for run in runs:
        spec = json.loads((run / "spec.json").read_text())
        level = ",".join(map(str, spec["levels"]))
        level = int(level) if level.isdigit() else level
        for f in sorted((run / "eval").glob("step_*.json")):
            jobs.append((run, level, f))
    with ProcessPoolExecutor(workers) as ex:
        out = [r for rows in ex.map(one_step, jobs) for r in rows]
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    num = [c for c in df.columns if c not in ("run", "level", "step", "id", "rule")]
    g = df.groupby(["level", "step"])[num]
    m, s = g.mean(), g.sem()
    out = m.add_suffix("_mean").join(s.add_suffix("_sem")).reset_index()
    return out


PANELS = [
    ("faith", "A   Faithfulness", "against the model's own choices"),
    ("tie", "B   Ties", "report true for both or neither option"),
    ("decisive_acc", "C   Accuracy when the report takes a side", ""),
    ("sign_ok", "D   Direction of relevant attributes", "same sign as the rule's attribute weight"),
    ("irr_mentioned", "E   Irrelevant attributes mentioned", "share the report depends on"),
    ("val_corr_avg", "F   Preference-vector match", "corr(avg report weights, rule weights)"),
]


def plot(summ: pd.DataFrame, path: Path, parse_floor: float = 0.25):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax, (key, title, sub) in zip(axes.flat, PANELS):
        _style(ax)
        for lv, s in summ.groupby("level"):
            s = s.sort_values("step")
            ok = s[s["parse_rate_mean"] >= parse_floor]
            if ok.empty or ok[f"{key}_mean"].isna().all():
                continue
            col = LEVEL_COLORS.get(lv, INK)
            m, e = ok[f"{key}_mean"], ok[f"{key}_sem"].fillna(0)
            ax.fill_between(ok.step, m - e, m + e, color=col, alpha=0.15, lw=0)
            ax.plot(ok.step, m, color=col, lw=2, marker="o", ms=3.5, markeredgecolor="white",
                    markeredgewidth=0.6)
        if key in ("faith", "decisive_acc", "sign_ok"):
            ax.axhline(0.5, color=MUTED, lw=1, ls=":")
        if key == "val_corr_avg":
            ax.axhline(0.0, color=MUTED, lw=1, ls=":")
        ax.set_title(title, color=INK, fontsize=11.5, loc="left", pad=18, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=8.5, va="bottom")
        ax.set_xlabel("Training step", color=MUTED)
    handles = [Line2D([], [], color=LEVEL_COLORS.get(lv, INK), lw=2, marker="o", ms=4, label=f"L{lv}")
               for lv in sorted(summ.level.unique(), key=str)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, -0.01))
    fig.text(0.5, -0.04, f"Checkpoints where under {parse_floor:.0%} of reports parse are left out. "
             "Dotted line: chance (A, C, D) or zero (F).", ha="center", color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, path: Path, ref_step: int):
    d = df[df.step == ref_step]
    cols = ["decision_acc", "parse_rate", "faith", "tie", "decisive_acc", "n_attr", "rel_mentioned",
            "irr_mentioned", "sign_ok", "val_corr_avg", "assoc_ceiling", "degree1_mass"]
    lines = [f"# Self-report diagnosis (step {ref_step})", "",
             "Means over characters.  `assoc_ceiling` is the best faithfulness a one-attribute "
             "report could reach; `degree1_mass` is the share of the rule explained by "
             "single-attribute weights.", "",
             "| level | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for lv, g in d.groupby("level"):
        lines.append(f"| L{lv} | " + " | ".join(f"{g[c].mean():.2f}" for c in cols) + " |")
    base = df[df.step == 0]
    if not base.empty:
        lines += ["", "## Untrained model (step 0)", "",
                  "| level | " + " | ".join(cols[1:10]) + " |", "|" + "---|" * 10]
        for lv, g in base.groupby("level"):
            lines.append(f"| L{lv} | " + " | ".join(f"{g[c].mean():.2f}" for c in cols[1:10]) + " |")
    if "obs_faith" in d:
        oc = ["obs_parse_rate", "obs_faith", "obs_tie", "obs_decisive_acc", "obs_irr_mentioned",
              "obs_sign_ok", "obs_val_corr_avg"]
        lines += ["", f"## Observer reports (step {ref_step})", "",
                  "| level | " + " | ".join(oc) + " |", "|" + "---|" * (len(oc) + 1)]
        for lv, g in d.groupby("level"):
            lines.append(f"| L{lv} | " + " | ".join(f"{g[c].mean():.2f}" for c in oc) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref-step", type=int, default=4000)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.runs, args.workers)
    df.to_csv(args.out / "per_character.csv", index=False)
    summ = summarize(df)
    summ.to_csv(args.out / "summary.csv", index=False)
    write_report(df, args.out / "report.md", args.ref_step)
    plot(summ, args.out / "diagnosis.png")
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
