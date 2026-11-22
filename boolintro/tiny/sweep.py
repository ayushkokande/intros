"""How many reports does pretraining need?  Class-2 results against N, the number of
class-1 characters whose rules were shown.

    python -m boolintro.tiny.sweep --runs runs/tiny/sweep_n*_s* --out runs/tiny_sweep

Each run is one (N, seed) point, made with `train pretrain --n-with-reports N` (all other
settings fixed, so the 1200 class-1 characters, their choices and the 200 class-2
characters are identical across N), then `train finetune --train embed`.

Faithfulness here counts an unparseable report as 0.5 (it says nothing either way), so
points with few valid reports (N = 0) stay on the same scale:
    faith_eff = parse_rate * faithfulness + (1 - parse_rate) * 0.5     (per character)

Outputs: sweep_chars.csv (per character), sweep.csv (per N, seed, family),
sweep_summary.md, sweep.png, and (when probes ran) sweep_probe.csv, sweep_probe.png.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..analyze import GRID, INK, LEVEL_COLORS, MUTED

FAM_COLORS = {"L1": LEVEL_COLORS[1], "L2": LEVEL_COLORS[2], "L3": LEVEL_COLORS[3], "L4": LEVEL_COLORS[4],
              "maj": "#7a5bd6", "mux": "#c2417a"}
FAM_LABEL = {"L1": "L1  one attribute", "L2": "L2  two attributes", "L3": "L3  2-term DNF",
             "L4": "L4  parity", "maj": "maj  2 of 3", "mux": "mux  if-then-else"}


def _eff(parse, faith):
    if parse is None or (isinstance(parse, float) and math.isnan(parse)):
        return math.nan
    f = 0.5 if faith is None or (isinstance(faith, float) and math.isnan(faith)) else faith
    return parse * f + (1 - parse) * 0.5


def load(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        spec = json.loads((run / "spec.json").read_text())
        n = spec.get("n_with_reports")
        if n is None:
            print(f"skip {run}: not a sweep run (no n_with_reports)")
            continue
        for ft in sorted(run.glob("ft_embed*")):
            path = ft / "eval_chars.jsonl"
            if not (ft / "DONE").exists() or not path.exists():
                print(f"skip {ft}: unfinished")
                continue
            recs = [json.loads(line) for line in path.read_text().splitlines()]
            last = max(r["step"] for r in recs)
            for r in recs:
                if r["step"] != last or r["group"] != "new":
                    continue
                rows.append({"run": run.name, "N": n, "seed": spec["seed"], "family": str(r["level"]),
                             "n_class1": sum(c["group"] != "new" for c in spec["characters"]),
                             "n_class2": sum(c["group"] == "new" for c in spec["characters"]),
                             "id": r["id"], "decision_acc": r["decision_acc"],
                             "parse_rate": r.get("formula_parse_rate"), "faith": r.get("formula_faith"),
                             "faith_eff": _eff(r.get("formula_parse_rate"), r.get("formula_faith")),
                             "faith_shuffled": r.get("formula_faith_shuffled"),
                             "exact": r.get("formula_exact")})
    return pd.DataFrame(rows)


def plot(per_seed: pd.DataFrame, path: Path, families: list[str], n1: int = 1200, n2: int = 200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    Ns = sorted(per_seed.N.unique())
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    for ax, key, title, sub in [
        (axes[0], "decision_acc", "A   Decision accuracy", "class-2 characters, held-out pairs"),
        (axes[1], "faith_eff", "B   Self-report faithfulness",
         "class-2 characters: does the reported rule match their own choices?")]:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.axhline(0.5, color=INK, lw=1.1, ls="--")
        ends = []
        for f in families:
            d = per_seed[per_seed.family == f].groupby("N")[key]
            m, e = d.mean().reindex(Ns), d.std().reindex(Ns).fillna(0)
            col = FAM_COLORS.get(f, INK)
            ax.fill_between(Ns, m - e, m + e, color=col, alpha=0.15, lw=0)
            ax.plot(Ns, m.values, color=col, lw=2.2, marker="o", ms=5, markeredgecolor="white",
                    markeredgewidth=0.8)
            ends.append((m.values[-1], f, col))
        merged = {}
        for y, f, col in ends:
            merged.setdefault(round(y, 2), []).append((f, col))
        ends = sorted((y, ", ".join(f for f, _ in v), v[0][1]) for y, v in merged.items())
        ys = [y for y, _, _ in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 0.035)
        for (y, lab, col), ly in zip(ends, ys):
            ax.annotate(f"{lab}  {y:.2f}", xy=(Ns[-1], y), xytext=(Ns[-1] * 1.45, ly), color=INK,
                        fontsize=9.5, va="center", annotation_clip=False,
                        arrowprops=dict(arrowstyle="-", color=col, lw=1, shrinkA=0, shrinkB=2))
        ax.set_xscale("symlog", linthresh=max(1, min(n for n in Ns if n > 0) if any(Ns) else 1))
        ax.set_xticks(Ns, [str(n) for n in Ns])
        ax.minorticks_off()
        ax.set_xlim(-0.3 if 0 in Ns else None, Ns[-1] * 1.15)
        ax.set_ylim(0.4, 1.05)
        ax.set_xlabel("class-1 characters whose rules were shown in pretraining (N)", color=MUTED)
        ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=22, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9.5)
    axes[0].set_ylabel("mean over class-2 characters (0.5 = chance)", color=MUTED)
    handles = [Line2D([], [], color=FAM_COLORS.get(f, INK), lw=2.2, marker="o", ms=5,
                      label=FAM_LABEL.get(f, f)) for f in families]
    handles.append(Line2D([], [], color=INK, lw=1.1, ls="--", label="chance"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9.5,
               bbox_to_anchor=(0.5, -0.03))
    fig.text(0.01, 0.99, "How many reports does pretraining need?", fontsize=15, fontweight="bold",
             color=INK, va="top")
    fig.tight_layout(rect=(0, 0.05, 0.95, 0.9))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def load_probe(runs: list[Path]) -> pd.DataFrame:
    """Per (N, seed): best-layer cosine of the P (degree-1) probe, mean over rule types with a P.
    class1 = cross-validated on class-1 characters (count-weighted over pre_report/pre_choice);
    class2 = probe fit on class 1, applied to the finetuned class-2 characters."""
    rows = []
    for run in runs:
        spec = json.loads((run / "spec.json").read_text())
        n, path = spec.get("n_with_reports"), run / "probe" / "probe.csv"
        if n is None or not path.exists():
            continue
        d = pd.read_csv(path)
        d = d[(d.target == "deg1") & (d.metric == "cos") & (d.family != "L4")]
        ft = d.model.str.replace(r"_s\d+$", "", regex=True)
        best = lambda x: (x.groupby(["family", "rep"]).value.mean().groupby("family").max().mean())  # noqa: E731
        counts = pd.Series([c["group"] for c in spec["characters"]]).value_counts()
        c1 = [(best(d[(ft == "pretrain") & (d.group == g)]), counts.get(g, 0))
              for g in ("pre_report", "pre_choice") if ((ft == "pretrain") & (d.group == g)).any()]
        c2 = d[(ft == "ft_embed") & (d.group == "new")]
        rows.append({"N": n, "seed": spec["seed"],
                     "class1": sum(v * k for v, k in c1) / sum(k for _, k in c1) if c1 else math.nan,
                     "class2": best(c2) if not c2.empty else math.nan})
    return pd.DataFrame(rows)


def plot_probe(probe: pd.DataFrame, per_seed: pd.DataFrame, path: Path, families=("L2", "L3")):
    """Left: is P stored (class 1) and are new people stored the same way (class 2)?
    Right: class-2 report faithfulness."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ns = sorted(set(probe.N) | set(per_seed.N))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    left = [("class1", "#2a78d6", "class-1 people:\nP readable"),
            ("class2", "#c2417a", "class-2 people:\nsame layout as class 1")]
    right = [(f, FAM_COLORS[f], FAM_LABEL[f].split("  ")[0]) for f in families]
    panels = [(axes[0], probe, left, None, "A   Can P be read from each person's vector?",
               "match with true P, using a reading formula fit on class 1", (0, 1.05), "match (1 = perfect)"),
              (axes[1], per_seed, right, "faith_eff", "B   Can the model write new people's rules?",
               "class-2 report faithfulness", (0.4, 1.0), "faithfulness (0.5 = chance)")]
    for ax, data, series, val, title, sub, ylim, ylab in panels:
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=10)
        ax.grid(axis="y", color=GRID, lw=0.8)
        for key, col, lab in series:
            if val is None:
                m = data.groupby("N")[key].mean().reindex(Ns)
            else:
                m = data[data.family == key].groupby("N")[val].mean().reindex(Ns)
            ax.plot(Ns, m.values, color=col, lw=2.4, marker="o", ms=6, markeredgecolor="white",
                    markeredgewidth=0.8)
            ax.annotate(lab, xy=(Ns[-1], m.values[-1]), xytext=(10, 0), textcoords="offset points",
                        color=col, fontsize=10, fontweight="bold", va="center", annotation_clip=False)
        if val is not None:
            ax.axhline(0.5, color=INK, lw=1.1, ls="--")
            ax.text(Ns[-1], 0.49, "chance", color=INK, fontsize=9, va="top", ha="right")
        ax.set_xscale("symlog", linthresh=max(1, min(n for n in Ns if n > 0)))
        ax.set_xticks(Ns, [str(n) for n in Ns])
        ax.minorticks_off()
        ax.set_xlim(-0.3 if 0 in Ns else None, Ns[-1] * 1.15)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab, color=MUTED)
        ax.set_xlabel("class-1 characters whose rules were shown in pretraining (N)", color=MUTED)
        ax.set_title(title, color=INK, fontsize=12.5, loc="left", pad=22, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9.5)
    fig.text(0.01, 0.99, "P is always stored; reports decide whether new people can be read",
             fontsize=15, fontweight="bold", color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 0.97, 0.92), w_pad=3)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--families", default="L1,L2,L3,L4", help="families to draw (all are in the CSVs)")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.runs)
    if df.empty:
        raise SystemExit("no finished sweep runs found")
    df.to_csv(args.out / "sweep_chars.csv", index=False)
    per_seed = (df.groupby(["N", "seed", "family"])
                [["decision_acc", "parse_rate", "faith", "faith_eff", "faith_shuffled", "exact"]]
                .mean().reset_index())
    per_seed.to_csv(args.out / "sweep.csv", index=False)
    fams = [f for f in args.families.split(",") if f in set(df.family)]
    plot(per_seed, args.out / "sweep.png", fams, int(df.n_class1.iloc[0]), int(df.n_class2.iloc[0]))
    probe = load_probe(args.runs)
    if not probe.empty:
        probe.to_csv(args.out / "sweep_probe.csv", index=False)
        plot_probe(probe, per_seed, args.out / "sweep_probe.png")
    g = per_seed.groupby(["family", "N"])
    t = g[["decision_acc", "parse_rate", "faith_eff"]].mean().round(2)
    t["faith_eff_sd"] = g.faith_eff.std().round(2)
    t["n_seeds"] = g.size()
    md = ("# How many reports does pretraining need?\n\nClass-2 results (end of finetuning), mean over "
          "seeds.  `faith_eff` counts unparseable reports as 0.5.\n\n```\n" + t.to_string() + "\n```\n")
    (args.out / "sweep_summary.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
