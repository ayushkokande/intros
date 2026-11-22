"""Compare from-scratch runs across report worlds, rule families and character groups.

    python -m boolintro.tiny.analyze --runs runs/tiny/formula_s* runs/tiny/attribute_s* \
        --out runs/tiny_analysis

Rows come from the last eval of each stage:
    pre_report, pre_choice    end of pretraining
    new                       end of finetuning (one row per finetune subdir)
    new_before                finetune step 0 (fresh random embeddings: the floor)

Outputs: summary.csv, report.md, faithfulness.png, preference_vector.png,
finetune_trajectories.png, and (when probe/ exists) probe_summary.csv, probe.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..analyze import INK, MUTED, _style

GROUP_STYLE = {"pre_report": ("#2a78d6", "o", "pretrain, with reports"),
               "pre_choice": ("#eb6834", "s", "pretrain, choices only"),
               "new": ("#1baf7a", "D", "new, finetuned on choices"),
               "new_before": ("#a3a29b", "x", "new, before finetuning")}
FAM_ORDER = ["L1", "L2", "maj", "L3", "mux", "L4", "1", "2", "3", "4"]


def world_name(spec: dict) -> str:
    """Report format, plus the families that got reports when pretraining was restricted."""
    w = spec.get("report_format", "formula")
    rf = spec.get("report_families")
    if rf:
        w = f"{w}-reports-{'+'.join(rf)}"
    if spec.get("n_with_reports") is not None:
        w = f"{w}-N{spec['n_with_reports']}"
    return w


def _rows(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def load(runs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(final rows, finetune trajectory rows)."""
    final, traj = [], []
    for run in runs:
        spec = json.loads((run / "spec.json").read_text())
        meta = {"run": run.name, "world": world_name(spec), "seed": spec["seed"]}
        pre = _rows(run / "pretrain" / "eval.jsonl")
        if pre:
            last = max(r["step"] for r in pre)
            final += [{**meta, "finetune": "-", **r} for r in pre
                      if r["step"] == last and r["group"] != "new"]
        for ft in sorted(run.glob("ft_*")):
            rows = _rows(ft / "eval.jsonl")
            if not rows:
                continue
            last = max(r["step"] for r in rows)
            for r in rows:
                if r["group"] != "new":
                    continue
                traj.append({**meta, "finetune": ft.name, **r})
                if r["step"] == last:
                    final.append({**meta, "finetune": ft.name, **r})
                if r["step"] == 0:
                    final.append({**meta, "finetune": ft.name, **r, "group": "new_before"})
    f, t = pd.DataFrame(final), pd.DataFrame(traj)
    for d in (f, t):
        if not d.empty:
            d["level"] = d["level"].astype(str)
    return f, t


def _fam_sort(fams):
    return sorted(fams, key=lambda x: FAM_ORDER.index(x) if x in FAM_ORDER else 99)


def dot_panels(df: pd.DataFrame, key: str, path: Path, ylabel: str, chance: float | None,
               ceiling: bool = False, shuffled: bool = False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    panels = [(w, k) for w in sorted(df.world.unique())
              for k in ("formula", "attribute") if f"{k}_{key}" in df and df[df.world == w][f"{k}_{key}"].notna().any()]
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.6), sharey=True, squeeze=False)
    for ax, (w, k) in zip(axes[0], panels):
        _style(ax)
        d = df[df.world == w]
        fams = _fam_sort(d.level.unique())
        groups = [g for g in GROUP_STYLE if g in set(d.group)]
        width = 0.7 / max(1, len(groups))
        for gi, g in enumerate(groups):
            col, mk, _ = GROUP_STYLE[g]
            s = d[d.group == g].groupby("level")
            m, e = s[f"{k}_{key}"].mean(), s[f"{k}_{key}"].std().fillna(0)
            xs = [i - 0.35 + width * (gi + 0.5) for i in range(len(fams))]
            ax.errorbar(xs, [m.get(f, np.nan) for f in fams], yerr=[e.get(f, 0) for f in fams], fmt=mk,
                        color=col, ms=6, mec="white" if mk != "x" else col, capsize=0, lw=1.2)
            if shuffled and f"{k}_faith_shuffled" in d:
                sh = s[f"{k}_faith_shuffled"].mean()
                ax.scatter(xs, [sh.get(f, np.nan) for f in fams], marker="_", s=120, color=col, alpha=0.6)
        if ceiling and "assoc_ceiling" in d:
            c = d[d.group == "pre_report"].groupby("level")["assoc_ceiling"].mean()
            for i, f in enumerate(fams):
                if f in c:
                    ax.hlines(c[f], i - 0.42, i + 0.42, color=INK, lw=1, ls="--", alpha=0.6)
        if chance is not None:
            ax.axhline(chance, color=MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(fams)), fams)
        ax.set_title(f"{w} world: {k} reports", color=INK, fontsize=11.5, loc="left", fontweight="bold")
    axes[0][0].set_ylabel(ylabel, color=MUTED)
    handles = [Line2D([], [], color=GROUP_STYLE[g][0], marker=GROUP_STYLE[g][1], ls="", ms=6,
                      label=GROUP_STYLE[g][2]) for g in GROUP_STYLE if g in set(df.group)]
    if shuffled:
        handles.append(Line2D([], [], color=MUTED, marker="_", ls="", ms=10, label="shuffled-character baseline"))
    if ceiling:
        handles.append(Line2D([], [], color=INK, ls="--", lw=1, label="best one-attribute report"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 6), frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_traj(traj: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..analyze import LEVEL_COLORS
    cols = {"L1": LEVEL_COLORS[1], "L2": LEVEL_COLORS[2], "L3": LEVEL_COLORS[3], "L4": LEVEL_COLORS[4],
            "maj": "#7a5bd6", "mux": "#c2417a"}
    worlds = sorted(traj.world.unique())
    fig, axes = plt.subplots(len(worlds), 2, figsize=(12, 4.2 * len(worlds)), squeeze=False)
    for wi, w in enumerate(worlds):
        d = traj[(traj.world == w) & (traj.finetune.str.startswith("ft_embed"))]
        k = "attribute" if w.startswith("attribute") else "formula"
        for ax, (key, title) in zip(axes[wi], [("decision_acc", "Decision accuracy"),
                                               (f"{k}_faith", f"Faithfulness ({k} reports)")]):
            _style(ax)
            for f in _fam_sort(d.level.unique()):
                s = d[d.level == f].groupby("step")[key].mean()
                ax.plot(s.index, s.values, color=cols.get(f, INK), lw=2, label=f)
            ax.axhline(0.5, color=MUTED, lw=1, ls=":")
            ax.set_title(f"{w} world: {title}, new characters", color=INK, fontsize=11, loc="left",
                         fontweight="bold")
            ax.set_xlabel("finetune step (embeddings of new characters only)", color=MUTED)
        axes[wi][1].legend(frameon=False, fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def load_probes(runs: list[Path]) -> pd.DataFrame:
    out = []
    for run in runs:
        p = run / "probe" / "probe.csv"
        if p.exists():
            spec = json.loads((run / "spec.json").read_text())
            d = pd.read_csv(p)
            d["world"], d["seed"], d["run"] = world_name(spec), spec["seed"], run.name
            out.append(d)
    return pd.concat(out) if out else pd.DataFrame()


def plot_probe(pr: pd.DataFrame, path: Path):
    """Best-layer cosine between probed and true coefficients, per family, for the new
    characters (transfer) and the pretraining characters (cross-validated)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    d = pr[(pr.metric == "cos") & pr.target.isin(["deg1", "deg2", "deg3"])]
    d = d[~d.model.eq("pretrain") | d.group.isin(["pre_report", "pre_choice"])]
    d = d[d.model.eq("pretrain") | d.group.eq("new")]
    best = d.groupby(["world", "seed", "group", "family", "target", "rep"]).value.mean().reset_index()
    best = best.groupby(["world", "seed", "group", "family", "target"]).value.max().reset_index()
    worlds = sorted(best.world.unique())
    tcol = {"deg1": "#2a78d6", "deg2": "#eb6834", "deg3": "#1baf7a"}
    gmk = {"pre_report": "o", "pre_choice": "s", "new": "D"}
    fig, axes = plt.subplots(1, len(worlds), figsize=(6.5 * len(worlds), 4.4), squeeze=False, sharey=True)
    for ax, w in zip(axes[0], worlds):
        _style(ax)
        b = best[best.world == w]
        fams = _fam_sort(b.family.unique())
        for ti, t in enumerate(("deg1", "deg2", "deg3")):
            for gi, g in enumerate(("pre_report", "pre_choice", "new")):
                s = b[(b.target == t) & (b.group == g)].groupby("family").value.mean()
                xs = [i - 0.33 + 0.22 * ti + 0.06 * gi for i in range(len(fams))]
                ax.scatter(xs, [s.get(f, np.nan) for f in fams], color=tcol[t], marker=gmk[g], s=26,
                           edgecolor="white", linewidth=0.5)
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(fams)), fams)
        ax.set_title(f"{w} world: what a linear probe recovers", color=INK, fontsize=11, loc="left",
                     fontweight="bold")
    axes[0][0].set_ylabel("cosine(probed, true coefficients), best layer", color=MUTED)
    handles = [Line2D([], [], color=tcol[t], marker="o", ls="", label=lab) for t, lab in
               (("deg1", "degree 1 (P)"), ("deg2", "degree 2"), ("deg3", "degree 3"))]
    handles += [Line2D([], [], color=MUTED, marker=gmk[g], ls="", label=g.replace("_", " ")) for g in gmk]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.07))
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_report(df: pd.DataFrame, pr: pd.DataFrame, path: Path):
    lines = ["# From-scratch GPT-2: reports by world, family and group", "",
             "Means over seeds (and characters).  `faith` is faithfulness to the model's own choices; "
             "`shuf` the shuffled-character baseline; `val_corr` the match between the average "
             "report's attribute weights and the rule's (the preference vector P).", ""]
    for w in sorted(df.world.unique()):
        for k in ("formula", "attribute"):
            if f"{k}_faith" not in df or df[df.world == w][f"{k}_faith"].isna().all():
                continue
            cols = ["decision_acc", f"{k}_faith", f"{k}_faith_shuffled", f"{k}_tie", f"{k}_decisive_acc",
                    f"{k}_sign_ok", f"{k}_irr_mentioned", f"{k}_val_corr_avg", f"{k}_exact", "assoc_ceiling"]
            d = df[df.world == w]
            t = d.groupby(["group", "level"])[cols].mean()
            t.columns = [c.replace(f"{k}_", "") for c in cols]
            lines += [f"## {w} world, {k} reports", "", "```", t.round(2).to_string(), "```", ""]
    if not pr.empty:
        b = (pr[(pr.metric == "r2") & (pr.family == "all")]
             .groupby(["world", "model", "group", "target", "rep"]).value.mean()
             .groupby(["world", "model", "group", "target"]).max().unstack("target"))
        lines += ["## Linear probes: best-layer R² (mean over seeds)", "", "```", b.round(2).to_string(), "```", ""]
    path.write_text("\n".join(lines))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df, traj = load(args.runs)
    df.to_csv(args.out / "summary.csv", index=False)
    pr = load_probes(args.runs)
    if not pr.empty:
        pr.to_csv(args.out / "probe_all.csv", index=False)
        plot_probe(pr, args.out / "probe.png")
    dot_panels(df, "faith", args.out / "faithfulness.png", "faithfulness (chance = 0.5)", 0.5,
               ceiling=True, shuffled=True)
    dot_panels(df, "val_corr_avg", args.out / "preference_vector.png",
               "corr(avg report weights, rule weights)", 0.0)
    if not traj.empty:
        plot_traj(traj, args.out / "finetune_trajectories.png")
    write_report(df, pr, args.out / "report.md")
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
