"""Summarise conflict-test results (boolintro.conflict) across runs.

    python -m boolintro.conflict_analyze --runs runs/qwen3-32b_L1 runs/qwen3-32b_L2 --out runs/conflict_analysis

Reads <run>/conflict/step_*/results.jsonl (the latest step per run, or --step).

Outputs:
  per_context.csv   one row per (run, character, model, condition, k)
  summary.csv       means and SEM over characters
  contrasts.csv     paired contrasts with bootstrap 95% CIs
  report.md         the key tables
  conflict_shift.png   report and behaviour shift vs. record length, per level
  conflict_detect.png  P(yes) for genuine vs. forged records, per framing
  conflict_summary.png, conflict_detect_summary.png   one-k summaries (largest k) for reports

Key quantities (shift: +1 = follows R' from the record, -1 = follows R from the weights):
  record effect     shift(forged_X, k) - shift(genuine_self, k).  Same pairs and length; only the
                    recorded choices differ, so context-length and format effects cancel.
  resistance        record effect(base) - record effect(ft).  > 0: the record moves the
                    fine-tuned model less than it moves an outsider who only sees the record.
  level resistance  report_shift(base) - report_shift(ft) in the same context (where each ends
                    up; positive by construction at small k because ft starts at R).
  report - behavior report_shift_decisive(ft) - behavior_shift_hard(ft), where the decisive
                    report shift ignores ties.  < 0: the report holds to R more than the model's
                    own choices do; ~ 0: the report rides on behaviour.
  self-attribution  report_shift(forged_self) - report_shift(forged_other).
  detection         P(yes | genuine) - P(yes | forged).  > 0: the model tells its real record
                    from a forged one.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .analyze import GRID, INK, MUTED, _style
from .metrics import bootstrap_ci

METRICS = ["behavior_shift", "behavior_shift_hard", "parse_rate", "report_shift", "report_shift_decisive",
           "report_pick_R", "report_pick_alt", "report_tie", "faith_in_context", "exact_R", "exact_alt"]
COND_COLORS = {"genuine_self": "#8a8984", "forged_self": "#2a78d6", "forged_other": "#eb6834",
               "forged_unlabeled": "#1baf7a"}
COND_NAMES = {"genuine_self": "genuine record, \"you chose\"", "forged_self": "forged, \"you chose\"",
              "forged_other": "forged, \"another assistant chose\"", "forged_unlabeled": "forged, unlabeled"}


def find_results(run: Path, step: str | None) -> Path:
    dirs = sorted((run / "conflict").glob("step_*"))
    if step:
        dirs = [d for d in dirs if d.name == f"step_{int(step):06d}"]
    dirs = [d for d in dirs if (d / "results.jsonl").exists()]
    if not dirs:
        raise SystemExit(f"no conflict results in {run / 'conflict'}")
    return dirs[-1] / "results.jsonl"


def load(runs: list[Path], step: str | None):
    rows, det = [], []
    for run in runs:
        for line in find_results(run, step).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            base = {"run": run.name, "level": r["level"], "id": r["id"], "name": r["name"]}
            for model, res in r["models"].items():
                for key, m in res["contexts"].items():
                    cond, k = key.split("|")
                    row = {**base, "model": model, "condition": cond, "k": int(k),
                           **{x: m.get(x, math.nan) for x in METRICS}}
                    decisive = row["report_pick_alt"] + row["report_pick_R"]
                    row["report_shift_decisive"] = ((row["report_pick_alt"] - row["report_pick_R"]) / decisive
                                                    if decisive and not math.isnan(decisive) else math.nan)
                    rows.append(row)
                for key, p in res["detect"].items():
                    framing, kind, k = key.split("|")
                    det.append({**base, "model": model, "framing": framing, "kind": kind,
                                "k": int(k), "p_yes": p})
    return pd.DataFrame(rows), pd.DataFrame(det)


def with_none_at_each_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Copy the no-record rows (k = 0) into every condition, so each curve starts at k = 0."""
    none = df[df.condition == "none"]
    extra = [none.assign(condition=c) for c in COND_COLORS]
    return pd.concat([df[df.condition != "none"], *extra], ignore_index=True)


def paired(a: pd.Series, b: pd.Series) -> dict:
    d = (a - b).dropna()
    lo, hi = bootstrap_ci(d.tolist())
    return {"diff": float(d.mean()) if len(d) else math.nan, "ci_lo": lo, "ci_hi": hi, "n": int(len(d))}


def contrasts(df: pd.DataFrame, det: pd.DataFrame) -> pd.DataFrame:
    out = []
    idx = ["level", "id"]
    for (level, cond, k), sub in df[df.condition != "none"].groupby(["level", "condition", "k"]):
        ft = sub[sub.model == "ft"].set_index(idx)
        base = sub[sub.model == "base"].set_index(idx)
        if len(base):
            for metric in ("report_shift", "behavior_shift"):
                out.append({"level": level, "contrast": f"level resistance ({metric})", "condition": cond,
                            "k": k, **paired(base[metric], ft[metric].reindex(base.index))})
        out.append({"level": level, "contrast": "report - behavior (ft)", "condition": cond, "k": k,
                    **paired(ft["report_shift_decisive"], ft["behavior_shift_hard"])})
    # Record effect: forged minus genuine at the same k (same pairs, only the labels differ).
    for (level, model, k), sub in df[df.k > 0].groupby(["level", "model", "k"]):
        gen = sub[sub.condition == "genuine_self"].set_index("id")
        for cond in ("forged_self", "forged_other", "forged_unlabeled"):
            f = sub[sub.condition == cond].set_index("id")
            for metric in ("report_shift", "behavior_shift"):
                out.append({"level": level, "contrast": f"record effect {metric} ({model})", "condition": cond,
                            "k": k, **paired(f[metric], gen[metric].reindex(f.index))})
    eff = {}
    for (level, model, cond, k), sub in df[df.k > 0].groupby(["level", "model", "condition", "k"]):
        eff[(level, model, cond, k)] = sub.set_index("id")
    for (level, model, cond, k), f in eff.items():
        if model != "ft" or not cond.startswith("forged") or (level, "base", cond, k) not in eff:
            continue
        g_ft, f_b, g_b = eff[(level, "ft", "genuine_self", k)], eff[(level, "base", cond, k)], \
            eff[(level, "base", "genuine_self", k)]
        for metric in ("report_shift", "behavior_shift"):
            e_ft = f[metric] - g_ft[metric].reindex(f.index)
            e_b = f_b[metric] - g_b[metric].reindex(f_b.index)
            out.append({"level": level, "contrast": f"resistance ({metric})", "condition": cond, "k": k,
                        **paired(e_b, e_ft.reindex(e_b.index))})
    for (level, model, k), sub in df.groupby(["level", "model", "k"]):
        s = sub[sub.condition == "forged_self"].set_index("id")
        o = sub[sub.condition == "forged_other"].set_index("id")
        if len(s) and len(o):
            out.append({"level": level, "contrast": f"self-attribution ({model})", "condition": "", "k": k,
                        **paired(s["report_shift"], o["report_shift"].reindex(s.index))})
    for (level, model, framing, k), sub in det.groupby(["level", "model", "framing", "k"]):
        g = sub[sub.kind == "genuine"].set_index("id")
        f = sub[sub.kind == "forged"].set_index("id")
        out.append({"level": level, "contrast": f"detection {framing} ({model})", "condition": "", "k": k,
                    **paired(g["p_yes"], f["p_yes"].reindex(g.index))})
    return pd.DataFrame(out)


def fmt(r) -> str:
    return f"{r['diff']:+.3f} [{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"


def report_md(df, con, det) -> str:
    lines = ["# Conflict test: weights vs. forged self-history", "",
             "Shift: +1 = follows the record's rule R', -1 = follows the trained rule R. "
             "Scored on held-out pairs where R and R' disagree. Means over characters; "
             "95% paired bootstrap CIs. ft = fine-tuned, base = adapter off (outsider).", ""]
    for level, sub in df.groupby("level"):
        n = sub.id.nunique()
        c = con[con.level == level]

        def get(contrast, cond, k):
            r = c[(c.contrast == contrast) & (c.condition == cond) & (c.k == k)]
            return fmt(r.iloc[0]) if len(r) else "–"
        lines += [f"## Level {level} ({n} characters)", "", "### Shift in each context", "",
                  "| condition | k | ft report | base report | ft behaviour | base behaviour | "
                  "ft report - behaviour (decisive) | ft faith in context | ft parse rate |",
                  "|" + "---|" * 9]
        for (cond, k), g in sub.groupby(["condition", "k"], sort=False):
            ft, base = g[g.model == "ft"], g[g.model == "base"]
            lines.append(
                f"| {cond} | {k} | {ft.report_shift.mean():+.3f} | {base.report_shift.mean():+.3f} | "
                f"{ft.behavior_shift.mean():+.3f} | {base.behavior_shift.mean():+.3f} | "
                f"{get('report - behavior (ft)', cond, k)} | {ft.faith_in_context.mean():.3f} | "
                f"{ft.parse_rate.mean():.2f} |")
        lines += ["", "### Record effect (forged minus genuine record, same pairs) and resistance", "",
                  "| condition | k | ft report effect | base report effect | resistance (report) | "
                  "ft behaviour effect | base behaviour effect | resistance (behaviour) |",
                  "|" + "---|" * 8]
        for cond in ("forged_self", "forged_other", "forged_unlabeled"):
            for k in sorted(sub[sub.k > 0].k.unique()):
                lines.append(
                    f"| {cond} | {k} | {get('record effect report_shift (ft)', cond, k)} | "
                    f"{get('record effect report_shift (base)', cond, k)} | {get('resistance (report_shift)', cond, k)} | "
                    f"{get('record effect behavior_shift (ft)', cond, k)} | "
                    f"{get('record effect behavior_shift (base)', cond, k)} | "
                    f"{get('resistance (behavior_shift)', cond, k)} |")
        lines += ["", "### Self-attribution and detection", "",
                  "| contrast | k | difference [95% CI] | n |", "|---|---|---|---|"]
        for r in c[c.condition == ""].sort_values(["contrast", "k"]).to_dict("records"):
            lines.append(f"| {r['contrast']} | {r['k']} | {fmt(r)} | {r['n']} |")
        d = det[det.level == level]
        if len(d):
            lines += ["", "Mean P(yes) by record type:", "",
                      "| model | framing | k | genuine | forged |", "|---|---|---|---|---|"]
            for (model, framing, k), g in d.groupby(["model", "framing", "k"]):
                lines.append(f"| {model} | {framing} | {k} | {g[g.kind == 'genuine'].p_yes.mean():.3f} | "
                             f"{g[g.kind == 'forged'].p_yes.mean():.3f} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def plot_shift(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = with_none_at_each_condition(df)
    levels = sorted(d.level.unique())
    fig, axes = plt.subplots(len(levels), 2, figsize=(11, 3.9 * len(levels)), sharey=True, squeeze=False)
    for row, level in zip(axes, levels):
        for ax, (metric, title) in zip(row, [("report_shift", "Self-report shift"),
                                             ("behavior_shift", "Behaviour shift")]):
            _style(ax)
            ax.axhline(0, color=MUTED, lw=1, ls=":")
            sub = d[d.level == level]
            for cond, col in COND_COLORS.items():
                for model, ls in (("ft", "-"), ("base", "--")):
                    g = sub[(sub.condition == cond) & (sub.model == model)].groupby("k")[metric]
                    if g.ngroups == 0:
                        continue
                    m, e = g.mean(), g.sem().fillna(0)
                    ax.plot(m.index, m.values, color=col, ls=ls, lw=2 if model == "ft" else 1.4,
                            marker="o" if model == "ft" else None, ms=4)
                    if model == "ft":
                        ax.fill_between(m.index, m - e, m + e, color=col, alpha=0.12, lw=0)
            ax.set_ylim(-1.22, 1.22)
            ax.set_xticks(sorted(sub.k.unique()))
            ax.set_xlabel("Records in context (k)", color=MUTED)
            ax.set_title(f"L{level}: {title}", color=INK, fontsize=11, loc="left", fontweight="bold")
            ax.text(0.01, 0.98, "follows record rule R'", transform=ax.transAxes, color=MUTED,
                    fontsize=8, va="top")
            ax.text(0.01, 0.02, "follows trained rule R", transform=ax.transAxes, color=MUTED,
                    fontsize=8, va="bottom")
        row[0].set_ylabel("Shift (+1 = R', -1 = R)", color=MUTED)
        row[1].tick_params(labelleft=True)
    handles = [Line2D([], [], color=c, lw=2, label=COND_NAMES[k]) for k, c in COND_COLORS.items()]
    handles += [Line2D([], [], color=INK, lw=2, marker="o", ms=4, label="fine-tuned"),
                Line2D([], [], color=INK, lw=1.4, ls="--", label="base model (outsider)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13 if len(levels) == 1 else 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_detect(det: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    levels = sorted(det.level.unique())
    fig, axes = plt.subplots(len(levels), 2, figsize=(10, 3.4 * len(levels)), sharey=True, squeeze=False)
    for row, level in zip(axes, levels):
        for ax, framing in zip(row, ("self", "third")):
            _style(ax)
            sub = det[(det.level == level) & (det.framing == framing)]
            for kind, col in (("genuine", "#8a8984"), ("forged", "#2a78d6")):
                for model, ls in (("ft", "-"), ("base", "--")):
                    g = sub[(sub.kind == kind) & (sub.model == model)].groupby("k").p_yes
                    if g.ngroups == 0:
                        continue
                    ax.plot(g.mean().index, g.mean().values, color=col, ls=ls,
                            lw=2 if model == "ft" else 1.4, marker="o" if model == "ft" else None, ms=4)
            ax.set_ylim(0, 1)
            ax.set_xticks(sorted(sub.k.unique()))
            q = "\"Did you make these decisions?\"" if framing == "self" else "\"Consistent with how <name> chooses?\""
            ax.set_title(f"L{level}: {q}", color=INK, fontsize=10, loc="left")
            ax.set_xlabel("Records shown (k)", color=MUTED)
        row[0].set_ylabel("P(yes)", color=MUTED)
        row[1].tick_params(labelleft=True)
    handles = [Line2D([], [], color="#8a8984", lw=2, label="genuine record"),
               Line2D([], [], color="#2a78d6", lw=2, label="forged record"),
               Line2D([], [], color=INK, lw=2, marker="o", ms=4, label="fine-tuned"),
               Line2D([], [], color=INK, lw=1.4, ls="--", label="base model")]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.13 if len(levels) == 1 else 0.06, 1, 1))
    fig.savefig(path, dpi=200)
    plt.close(fig)


FAKE_LABELS = {"forged_self": "fake\n\"you chose\"", "forged_other": "fake \"another\nassistant chose\"",
               "forged_unlabeled": "fake, plain\nexamples"}


def plot_summary(con: pd.DataFrame, path: Path, k: int):
    """Record effect (fake minus genuine memory) at one k: trained report, trained choices, outsider report."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    series = [("record effect report_shift (ft)", "Trained model: self-report", "#2a78d6", "o", -0.22),
              ("record effect behavior_shift (ft)", "Trained model: choices", "#eb6834", "s", 0.0),
              ("record effect report_shift (base)", "Outsider: self-report", "#8a8984", "D", 0.22)]
    levels = sorted(con.level.unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(5.4 * len(levels), 4.2), sharey=True, squeeze=False)
    for ax, level in zip(axes[0], levels):
        _style(ax)
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        conds = list(FAKE_LABELS)
        for name, label, col, mk, dx in series:
            for i, cond in enumerate(conds):
                r = con[(con.level == level) & (con.contrast == name) & (con.condition == cond) & (con.k == k)]
                if not len(r):
                    continue
                r = r.iloc[0]
                ax.errorbar(i + dx, r["diff"], yerr=[[r["diff"] - r["ci_lo"]], [r["ci_hi"] - r["diff"]]],
                            fmt=mk, color=col, ms=7, capsize=3, lw=1.5, markeredgecolor="white")
                ax.text(i + dx, r["ci_hi"] + 0.02, f"{r['diff']:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
        ax.set_xticks(range(len(conds)), [FAKE_LABELS[c] for c in conds], fontsize=9)
        ax.set_xlim(-0.6, len(conds) - 0.4)
        ax.set_title(f"Level {level}", color=INK, fontsize=12, loc="left", fontweight="bold")
    axes[0][0].set_ylabel("Pull toward the fake rule\n(fake minus genuine memory)", color=MUTED)
    axes[0][0].set_ylim(-0.15, 0.65)
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=True)
    handles = [Line2D([], [], color=c, marker=m, ls="none", ms=7, label=l) for _, l, c, m, _ in series]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_detect_summary(det: pd.DataFrame, path: Path, k: int):
    """P(yes) for genuine vs. fake memories at one k, per question and model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups = [("ft", "self", "Trained:\n\"Did you make\nthese?\""),
              ("ft", "third", "Trained:\n\"Consistent with\nhow <name> chooses?\""),
              ("base", "self", "Outsider:\n\"Did you make\nthese?\""),
              ("base", "third", "Outsider:\n\"Consistent with\nhow <name> chooses?\"")]
    levels = sorted(det.level.unique())
    fig, axes = plt.subplots(1, len(levels), figsize=(5.6 * len(levels), 4.2), sharey=True, squeeze=False)
    w = 0.36
    for ax, level in zip(axes[0], levels):
        _style(ax)
        sub = det[(det.level == level) & (det.k == k)]
        for i, (model, framing, _) in enumerate(groups):
            g = sub[(sub.model == model) & (sub.framing == framing)]
            for j, (kind, col) in enumerate((("genuine", "#8a8984"), ("forged", "#2a78d6"))):
                v = g[g.kind == kind].p_yes
                m, e = v.mean(), v.sem()
                x = i + (j - 0.5) * w
                ax.bar(x, m, w * 0.92, color=col, yerr=e, capsize=2, ecolor=MUTED)
                ax.text(x, m + (e if e == e else 0) + 0.02, f"{m:.2f}", ha="center", va="bottom",
                        fontsize=8, color=INK)
        ax.set_xticks(range(len(groups)), [g[2] for g in groups], fontsize=8)
        ax.set_ylim(0, 1.08)
        ax.set_title(f"Level {level}", color=INK, fontsize=12, loc="left", fontweight="bold")
    axes[0][0].set_ylabel("P(yes)", color=MUTED)
    for ax in axes[0][1:]:
        ax.tick_params(labelleft=True)
    fig.legend(handles=[Patch(color="#8a8984", label="genuine memory"), Patch(color="#2a78d6", label="fake memory")],
               loc="lower center", ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--step", default=None)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    df, det = load(args.runs, args.step)
    df.to_csv(args.out / "per_context.csv", index=False)
    g = df.groupby(["level", "model", "condition", "k"])[METRICS]
    pd.concat([g.mean().add_suffix("_mean"), g.sem().add_suffix("_sem"),
               g.size().rename("n_characters")], axis=1).reset_index().to_csv(args.out / "summary.csv", index=False)
    con = contrasts(df, det)
    con.to_csv(args.out / "contrasts.csv", index=False)
    md = report_md(df, con, det)
    (args.out / "report.md").write_text(md)
    print(md)
    if not args.no_plots:
        plot_shift(df, args.out / "conflict_shift.png")
        kmax = int(df.k.max())
        plot_summary(con, args.out / "conflict_summary.png", kmax)
        if len(det):
            plot_detect(det, args.out / "conflict_detect.png")
            plot_detect_summary(det, args.out / "conflict_detect_summary.png", kmax)
        print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
