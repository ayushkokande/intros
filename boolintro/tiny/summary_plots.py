"""Four simple summary figures, one per finding.

    python -m boolintro.tiny.summary_plots --analysis runs/tiny_analysis \
        --diagnosis runs/qwen3-32b_diagnosis --out runs/tiny_analysis/summary_figs

    fig1_P_is_stored.png            a linear readout recovers P for every rule type
    fig1_P_is_stored_formula.png    the same, formula world only
    fig2_structure_needs_words.png  interactions are linearly readable only in the formula world
    fig3_new_characters.png         report faithfulness for new, choices-only characters
    fig4_qwen_L3.png                Qwen L3: knows the directions, keeps the irrelevant attributes
    fig0_gpt2_experiment.png        the professor's GPT-2 experiment, drawn like the original trajectories
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..analyze import GRID, INK, MUTED

WORLD = {"formula": ("#2a78d6", "Formula world\n(pretraining reports = full rules)"),
         "attribute": ("#eb6834", "Attribute world\n(pretraining reports = 'cares about x')")}
FAMS = ["L1", "L2", "maj", "L3", "mux", "L4"]
FAM_LABEL = {"L1": "L1\none attribute", "L2": "L2\ntwo attributes", "maj": "maj\n2 of 3",
             "L3": "L3\n(a AND b) OR\n(c AND d)", "mux": "mux\nif a then b\nelse c", "L4": "L4\nparity"}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": INK, "ytick.color": MUTED, "figure.facecolor": "white"})
    return plt


def _clean(ax):
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _title(fig, title, sub):
    fig.text(0.01, 0.99, title, fontsize=15, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.925, sub, fontsize=11, color=MUTED, va="top")


def _bars(ax, groups, series, values, errs=None, fmt="{:.2f}", ymax=1.0):
    """Grouped bars; series = [(key, colour, label)], values[key][group]."""
    n = len(series)
    w = 0.8 / n
    for si, (key, col, lab) in enumerate(series):
        xs = np.arange(len(groups)) - 0.4 + w * (si + 0.5)
        ys = [values[key].get(g, np.nan) for g in groups]
        es = [errs[key].get(g, 0) for g in groups] if errs else None
        ax.bar(xs, ys, w - 0.04, color=col, label=lab, yerr=es,
               error_kw=dict(ecolor=MUTED, lw=1, capsize=0))
        for x, y, e in zip(xs, ys, es or [0] * len(ys)):
            if not np.isnan(y):
                ax.text(x, max(y, 0) + (e or 0) + ymax * 0.015, fmt.format(y), ha="center", va="bottom",
                        fontsize=9.5, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_ylim(0, ymax * 1.08)


def best_cos(pr: pd.DataFrame) -> pd.DataFrame:
    """Best-layer cosine per (world, seed, model, group, family, target), mean over seeds."""
    c = pr[pr.metric == "cos"]
    b = (c.groupby(["world", "seed", "model", "group", "family", "target", "rep"]).value.mean()
         .groupby(["world", "seed", "model", "group", "family", "target"]).max().reset_index())
    b["ft"] = b.model.str.replace(r"_s\d+$", "", regex=True)
    return b


def fig1(b: pd.DataFrame, path: Path, worlds=("formula", "attribute")):
    plt = _plt()
    d = b[(b.ft == "pretrain") & b.group.isin(["pre_report", "pre_choice"]) & (b.target == "deg1")]
    fams = [f for f in FAMS if f != "L4"]
    vals = {w: d[d.world == w].groupby("family").value.mean().to_dict() for w in worlds}
    one = len(worlds) == 1
    fig, ax = plt.subplots(figsize=(9 if one else 10, 5.2))
    _clean(ax)
    _bars(ax, fams, [(w, WORLD[w][0], WORLD[w][1].replace("\n", " ")) for w in worlds], vals)
    ax.set_xticklabels([FAM_LABEL[f] for f in fams], fontsize=9.5)
    ax.set_ylabel("match with true P (1 = perfect)")
    if one:  # a single series needs no legend; the subtitle names the world
        sub = ("A simple linear read-out of each character's representation recovers its P almost\n"
               "perfectly, for every rule type.  (Pretrained characters, tested on ones the probe was\n"
               "not fit on; parity has no P.)")
    else:
        ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2, fontsize=10)
        sub = ("A simple linear read-out of each character's representation recovers its P almost perfectly,\n"
               "for every rule type, in both worlds.  (Pretrained characters; parity has no P.)")
    _title(fig, "Each character's preference vector P can be read out linearly" if one else
           "1.  Each character's preference vector P can be read out linearly", sub)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig2(b: pd.DataFrame, path: Path):
    plt = _plt()
    d = b[(b.ft == "pretrain") & b.group.isin(["pre_report", "pre_choice"])]
    parts = {
        "P\n(one weight per attribute)": d[(d.target == "deg1") & (d.family != "L4")],
        "Pairwise interactions\n(AND / if-then in L2, L3, mux)": d[(d.target == "deg2") & d.family.isin(["L2", "L3", "mux"])],
        "Three-way interaction\n(parity, L4)": d[(d.target == "deg3") & (d.family == "L4")],
    }
    vals = {w: {k: v[v.world == w].value.mean() for k, v in parts.items()} for w in WORLD}
    fig, ax = plt.subplots(figsize=(10, 5.2))
    _clean(ax)
    groups = list(parts)
    _bars(ax, groups, [(w, WORLD[w][0], WORLD[w][1].replace("\n", " ")) for w in WORLD], vals)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("match with the truth (1 = perfect)")
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=10)
    _title(fig, "2.  The rule's structure is readable mainly when the model had to put it into words",
           "Decisions are learned in both worlds (parity only partly in the attribute world).  But the AND / if-then /\n"
           "parity parts are linearly readable mainly in the formula world, where pretraining asked to write them out.")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig3(summ: pd.DataFrame, path: Path):
    plt = _plt()
    d = summ[(summ.group == "new") & summ.finetune.str.startswith("ft_embed")].copy()
    d["faith"] = np.where(d.world == "formula", d["formula_faith"], d["attribute_faith"])
    per_seed = d.groupby(["world", "seed", "level"]).faith.mean().reset_index()
    vals = {w: per_seed[per_seed.world == w].groupby("level").faith.mean().to_dict() for w in WORLD}
    errs = {w: per_seed[per_seed.world == w].groupby("level").faith.std().to_dict() for w in WORLD}
    fig, ax = plt.subplots(figsize=(11, 5.4))
    _clean(ax)
    _bars(ax, FAMS, [(w, WORLD[w][0], WORLD[w][1].replace("\n", " ")) for w in WORLD], vals, errs)
    ax.set_ylim(0.4, 0.9)
    ax.axhline(0.5, color=INK, lw=1.2, ls="--")
    ax.set_xticklabels([FAM_LABEL[f] for f in FAMS], fontsize=9.5)
    ax.set_ylabel("report faithfulness")
    from matplotlib.lines import Line2D
    h, lab = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=1.2, ls="--"))
    lab.append("chance (0.5)")
    ax.legend(h, lab, frameon=False, loc="lower center", bbox_to_anchor=(0.5, -0.45), ncol=3, fontsize=10)
    _title(fig, "3.  New characters, learned from choices only: L3 is reported about as well as L1 and L2",
           "When pretraining taught full rules, new characters' reports beat chance about equally for L1, L2 and L3.\n"
           "When it taught only 'cares about x', reports are weaker for every rule.  (Bars: mean of 3 seeds; lines: spread.)")
    fig.tight_layout(rect=(0, 0, 1, 0.83))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig4(diag: pd.DataFrame, path: Path, step: int = 4000):
    plt = _plt()
    d = diag[(diag.step == step) & diag.level.isin([1, 2, 3])]
    m = d.groupby("level")[["val_corr_avg", "irr_mentioned", "faith"]].mean()
    lv_col = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
    panels = [("val_corr_avg", "Knows the direction of\neach attribute (P)", "match with true P (correlation)", (0, 1)),
              ("irr_mentioned", "Mentions attributes\nthat don't matter", "share of irrelevant attributes named", (0, 1)),
              ("faith", "Report matches its\nown choices", "faithfulness (0.5 = chance)", (0.4, 0.85))]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    for ax, (key, t, yl, lim) in zip(axes, panels):
        _clean(ax)
        ys = [m.loc[lv, key] for lv in (1, 2, 3)]
        ax.bar(range(3), [y - lim[0] for y in ys], 0.62, bottom=lim[0], color=[lv_col[lv] for lv in (1, 2, 3)])
        for i, y in enumerate(ys):
            ax.text(i, y + (lim[1] - lim[0]) * 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xticks(range(3), ["L1", "L2", "L3"])
        ax.set_ylim(*lim)
        if key == "faith":
            ax.axhline(0.5, color=INK, lw=1.2, ls="--")
        ax.set_title(t, fontsize=11.5, color=INK, loc="left", fontweight="bold")
        ax.set_ylabel(yl, fontsize=9.5)
    _title(fig, "4.  Qwen at L3: knows the directions, misses the structure",
           "Its L3 reports get most attribute directions right, but keep naming attributes that don't matter.\n"
           f"So they rarely pick a side, and faithfulness drops.  (Qwen3-32B, step {step}.)")
    fig.tight_layout(rect=(0, 0, 1, 0.80))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig0(runs: list[Path], path: Path):
    """The professor's experiment, drawn like the original trajectories figure: class-2
    (new, choices-only) characters during finetuning, one line per level (formula world)."""
    plt = _plt()
    from matplotlib.lines import Line2D

    from ..analyze import LEVEL_COLORS
    from .analyze import load
    _, traj = load(runs)
    d = traj[traj.finetune.str.startswith("ft_embed") & traj.level.isin(["L1", "L2", "L3", "L4"])]
    per_seed = d.groupby(["seed", "level", "step"])[["decision_acc", "formula_faith", "formula_faith_shuffled"]].mean()
    g = per_seed.groupby(["level", "step"])
    m, e = g.mean(), g.std()
    pre = traj.iloc[0:0]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    last = int(d.step.max())
    for ax, key, title, sub in [
        (axes[0], "decision_acc", "A   Decision accuracy", "held-out pairs, class-2 characters"),
        (axes[1], "formula_faith", "B   Self-report faithfulness",
         "does the reported rule match the model's own choices?")]:
        _clean(ax)
        ax.spines["left"].set_visible(True)
        ax.axhline(0.5, color=INK, lw=1.1, ls="--")
        ends = []
        for lv in ["L1", "L2", "L3", "L4"]:
            col = LEVEL_COLORS[int(lv[1])]
            mm, ee = m.loc[lv][key], e.loc[lv][key].fillna(0)
            ax.fill_between(mm.index, mm - ee, mm + ee, color=col, alpha=0.15, lw=0)
            ax.plot(mm.index, mm.values, color=col, lw=2.2, marker="o", ms=3.5, markeredgecolor="white",
                    markeredgewidth=0.6)
            if key == "formula_faith":
                sh = m.loc[lv]["formula_faith_shuffled"]
                ax.plot(sh.index, sh.values, color=col, lw=1, ls=":", alpha=0.8)
            ends.append((mm.values[-1], lv, col))
        # direct end labels: levels with the same rounded value share one label, then spread apart
        merged = {}
        for y, lv, col in ends:
            merged.setdefault(round(y, 2), []).append((lv, col))
        ends = [(y, ", ".join(l for l, _ in v), v[0][1]) for y, v in merged.items()]
        ends.sort()
        ys = [y for y, _, _ in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 0.035)
        for (y, lv, col), ly in zip(ends, ys):
            ax.annotate(f"{lv}  {y:.2f}", xy=(last, y), xytext=(last * 1.04, ly), color=INK, fontsize=9.5,
                        va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col, lw=1, shrinkA=0, shrinkB=2))
        ax.set_ylim(0.4, 1.05)
        ax.set_xlim(-100, last * 1.02)
        ax.set_xlabel("finetuning step (class 2: only their new embeddings are trained)")
        ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=22, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9.5)
    axes[0].set_ylabel("mean over class-2 characters (0.5 = chance)")
    handles = [Line2D([], [], color=LEVEL_COLORS[i], lw=2.2, marker="o", ms=4, label=lab) for i, lab in
               [(1, "L1  one attribute"), (2, "L2  two attributes"), (3, "L3  2-term DNF"), (4, "L4  parity")]]
    handles += [Line2D([], [], color=MUTED, lw=1, ls=":", label="shuffled-character baseline (B)"),
                Line2D([], [], color=INK, lw=1.1, ls="--", label="chance")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.03))
    _title(fig, "Small GPT-2 from scratch: can it report the rules of people it only saw choose?",
           "Pretrain on class 1 (choices + reports).  Then add class 2 with fresh random embeddings and train on their\n"
           "choices only.  Here, L3 reports are about as faithful as L1/L2.  (Mean of 3 seeds; band = spread.)")
    fig.tight_layout(rect=(0, 0.05, 0.97, 0.83))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--analysis", type=Path, default=Path("runs/tiny_analysis"))
    p.add_argument("--diagnosis", type=Path, default=Path("runs/qwen3-32b_diagnosis"))
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--tiny-runs", type=Path, default=Path("runs/tiny"), help="folder with formula_s* runs")
    args = p.parse_args(argv)
    out = args.out or args.analysis / "summary_figs"
    out.mkdir(parents=True, exist_ok=True)
    b = best_cos(pd.read_csv(args.analysis / "probe_all.csv"))
    fig1(b, out / "fig1_P_is_stored.png")
    fig1(b, out / "fig1_P_is_stored_formula.png", worlds=("formula",))
    fig2(b, out / "fig2_structure_needs_words.png")
    fig3(pd.read_csv(args.analysis / "summary.csv"), out / "fig3_new_characters.png")
    diag = args.diagnosis / "per_character.csv"
    if diag.exists():
        fig4(pd.read_csv(diag), out / "fig4_qwen_L3.png")
    fruns = sorted(args.tiny_runs.glob("formula_s*"))
    if fruns:
        fig0(fruns, out / "fig0_gpt2_experiment.png")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
