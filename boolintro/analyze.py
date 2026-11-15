"""Aggregate evaluation results across runs and checkpoints; write tables and figures.

    python -m boolintro.analyze --runs runs/L1 runs/L2 runs/L3 runs/L4 --out results/main

Outputs:
  per_character.csv     one row per (run, step, character)
  summary.csv           mean and SEM over characters per (run, level, step)
  report.md             final-checkpoint table with bootstrap CIs for the key contrasts
  trajectories.png      decision accuracy and faithfulness over training, per level
  faithfulness_dist.png per-character faithfulness at the final checkpoint
  self_vs_observer.png  self-report vs. observer faithfulness (if the observer ran)

Key contrasts (all paired over characters, final checkpoint):
  faithfulness - shuffled      > 0  reports are specific to the character's own behaviour
  faithfulness - base (step 0) > 0  training produced the faithful report
  error_faithfulness - 0.5     > 0  on pairs where behaviour departs from the target, the
                                    report follows behaviour (not the training label)
  faithfulness - observer      > 0  self-report beats an outside model that watches K decisions
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .metrics import bootstrap_ci

METRICS = ["decision_acc", "decision_p_target", "parse_rate", "faithfulness", "shuffled_faithfulness",
           "report_target_agreement", "exact_match", "report_tie_rate", "error_faithfulness",
           "n_error_pairs", "observer_faithfulness", "observer_target_agreement", "observer_parse_rate"]

# Categorical slots 1-4 of the reference palette, one per difficulty level (fixed order).
LEVEL_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a", 4: "#eda100"}
LEVEL_NAMES = {1: "L1 one literal", 2: "L2 AND/OR of two", 3: "L3 2-term DNF", 4: "L4 parity of three"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"


def load(runs: list[Path]) -> pd.DataFrame:
    rows = []
    for run in runs:
        for f in sorted((run / "eval").glob("step_*.json")):
            res = json.loads(f.read_text())
            for m in res["per_character"]:
                rows.append({"run": run.name, "step": res["step"], **m})
    if not rows:
        raise SystemExit("no eval/step_*.json files found")
    df = pd.DataFrame(rows).drop(columns=["p_first"], errors="ignore")
    for col in METRICS:
        if col not in df:
            df[col] = np.nan
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["run", "level", "step"])
    out = g[METRICS].mean().add_suffix("_mean")
    sem = g[METRICS].sem().add_suffix("_sem")
    n = g.size().rename("n_characters")
    return pd.concat([out, sem, n], axis=1).reset_index()


def paired(a: pd.Series, b: pd.Series | float) -> dict:
    d = (a - b).dropna()  # b: per-character Series (aligned on id) or a constant
    lo, hi = bootstrap_ci(d.tolist())
    return {"diff": float(d.mean()) if len(d) else math.nan, "ci_lo": lo, "ci_hi": hi, "n": int(len(d))}


def final_contrasts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (run, level), sub in df.groupby(["run", "level"]):
        final = sub[sub.step == sub.step.max()].set_index("id")
        base = sub[sub.step == 0].set_index("id")
        row = {"run": run, "level": level, "final_step": int(sub.step.max())}
        for k in ["decision_acc", "faithfulness", "shuffled_faithfulness", "error_faithfulness",
                  "exact_match", "parse_rate", "observer_faithfulness"]:
            row[k] = float(final[k].mean())
        row["base_faithfulness"] = float(base["faithfulness"].mean()) if len(base) else math.nan
        for name, other in [("vs_shuffled", final["shuffled_faithfulness"]),
                            ("vs_base", base["faithfulness"].reindex(final.index) if len(base) else np.nan),
                            ("vs_observer", final["observer_faithfulness"])]:
            r = paired(final["faithfulness"], other)
            row.update({f"{name}_diff": r["diff"], f"{name}_ci": f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"})
        r = paired(final["error_faithfulness"], 0.5)
        row.update(error_vs_half_diff=r["diff"], error_vs_half_ci=f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]",
                   error_vs_half_n=r["n"])
        rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _dodge(ys: list[float], gap: float, top: float) -> list[float]:
    """Spread label positions so neighbours are at least `gap` apart and none is above `top`."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for a, b in zip(order, order[1:]):
        out[b] = max(out[b], out[a] + gap)
    out[order[-1]] = min(out[order[-1]], top)
    for a, b in zip(order[::-1], order[::-1][1:]):
        out[b] = min(out[b], out[a] - gap)
    return out


def plot_trajectories(summ: pd.DataFrame, path: Path, warmup: int | None = None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups = [(lv, s.sort_values("step")) for (_, lv), s in summ.groupby(["run", "level"])]
    last_step = int(summ.step.max())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.9), sharey=True)
    panels = [("decision_acc", "A   Decision accuracy", "model's choice matches the training target"),
              ("faithfulness", "B   Self-report faithfulness", "reported rule predicts the model's own choice")]
    for ax, (key, title, sub) in zip(axes, panels):
        _style(ax)
        ax.hlines(0.5, -last_step * 0.02, last_step, color=MUTED, lw=1, ls=":")
        ax.text(last_step * 0.01, 0.492, "chance", color=MUTED, fontsize=8, va="top")
        if warmup:
            ax.axvline(warmup, color=MUTED, lw=1, ls=(0, (2, 3)))
            ax.text(warmup + 60, 0.335, "LR warmup ends", color=MUTED, fontsize=8, va="bottom")
        ends = []
        for level, s in groups:
            col = LEVEL_COLORS.get(level, INK)
            m, e = s[f"{key}_mean"], s[f"{key}_sem"].fillna(0)
            if key == "faithfulness":
                ax.plot(s.step, s["shuffled_faithfulness_mean"], color=col, lw=1, ls="--", alpha=0.7)
            ax.fill_between(s.step, m - e, m + e, color=col, alpha=0.15, lw=0)
            ax.plot(s.step, m, color=col, lw=2, marker="o", ms=4,
                    markeredgecolor="white", markeredgewidth=0.8)
            ends.append((level, float(m.iloc[-1]), col))
        # Direct labels at the right end of every line, dodged so they don't collide.
        label_y = _dodge([y for _, y, _ in ends], gap=0.045, top=1.02)
        for (level, y, col), ly in zip(ends, label_y):
            ax.annotate(f"L{level} {y:.2f}", xy=(last_step, y), xytext=(last_step * 1.04, ly),
                        color=INK, fontsize=9, va="center",
                        arrowprops=dict(arrowstyle="-", color=col, lw=1, shrinkA=0, shrinkB=3))
        ax.set_xlim(-last_step * 0.02, last_step * 1.2)
        ax.set_xticks(range(0, last_step + 1, 1000))
        ax.tick_params(labelleft=True)
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=20, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
        ax.set_xlabel("Training step", color=MUTED)
    axes[0].set_ylabel("Mean over characters (chance = 0.5)", color=MUTED)
    axes[0].set_ylim(0.32, 1.04)
    handles = [Line2D([], [], color=LEVEL_COLORS.get(lv, INK), lw=2, marker="o", ms=4,
                      label=LEVEL_NAMES.get(lv, f"L{lv}")) for lv, _ in groups]
    handles.append(Line2D([], [], color=MUTED, lw=1, ls="--", label="shuffled-character baseline (B)"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=9,
               handlelength=2.2, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.07, 1, 1), w_pad=3)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_distribution(df: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = list(df.groupby(["run", "level"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.2), sharey=True, squeeze=False)
    bins = np.linspace(0, 1, 21)
    for ax, ((run, level), sub) in zip(axes[0], groups):
        _style(ax)
        final = sub[sub.step == sub.step.max()]
        ax.hist(final.faithfulness.dropna(), bins=bins, color=LEVEL_COLORS.get(level, INK),
                edgecolor="white", linewidth=1)
        ax.axvline(0.5, color=MUTED, lw=1, ls=":")
        ax.set_title(f"{LEVEL_NAMES.get(level, level)}\n{run}, step {int(sub.step.max())}",
                     fontsize=9, color=INK, loc="left")
        ax.set_xlabel("Per-character faithfulness", color=MUTED, fontsize=9)
    axes[0][0].set_ylabel("Characters", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_observer(df: pd.DataFrame, path: Path) -> bool:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    obs = df.dropna(subset=["observer_faithfulness"])
    if obs.empty:
        return False
    last = obs[obs.step == obs.groupby(["run", "level"]).step.transform("max")]
    g = last.groupby("level")[["faithfulness", "observer_faithfulness", "shuffled_faithfulness"]]
    mean, sem = g.mean(), g.sem().fillna(0)
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    _style(ax)
    x = np.arange(len(mean))
    styles = [("faithfulness", "Self-report", 1.0), ("observer_faithfulness", "Observer (base model + K decisions)", 0.55),
              ("shuffled_faithfulness", "Shuffled baseline", 0.25)]
    w = 0.26
    for k, (col, lab, alpha) in enumerate(styles):
        colors = [LEVEL_COLORS.get(lv, INK) for lv in mean.index]
        ax.bar(x + (k - 1) * w, mean[col], w * 0.92, yerr=sem[col], color=colors, alpha=alpha,
               edgecolor=INK if k == 0 else "none", linewidth=0.6, capsize=2, ecolor=MUTED, label=lab)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xticks(x, [LEVEL_NAMES.get(lv, lv) for lv in mean.index], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Faithfulness to model's choices", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    df = load(args.runs)
    df.drop(columns=[c for c in ("rule",) if c in df]).to_csv(args.out / "per_character.csv", index=False)
    summ = summarize(df)
    summ.to_csv(args.out / "summary.csv", index=False)
    con = final_contrasts(df)
    con.to_csv(args.out / "final_contrasts.csv", index=False)

    lines = ["# Boolean introspection results", "",
             "Final checkpoint per run and level; means over characters; 95% paired bootstrap CIs.", "",
             "| run | level | step | decision acc | faithfulness | shuffled | base (step 0) | "
             "faith - shuffled | faith - base | error-pair faith - 0.5 | exact match | observer | faith - observer |",
             "|" + "---|" * 13]
    for r in con.itertuples():
        lines.append(
            f"| {r.run} | {r.level} | {r.final_step} | {r.decision_acc:.3f} | {r.faithfulness:.3f} | "
            f"{r.shuffled_faithfulness:.3f} | {r.base_faithfulness:.3f} | {r.vs_shuffled_diff:+.3f} {r.vs_shuffled_ci} | "
            f"{r.vs_base_diff:+.3f} {r.vs_base_ci} | {r.error_vs_half_diff:+.3f} {r.error_vs_half_ci} (n={r.error_vs_half_n}) | "
            f"{r.exact_match:.3f} | {r.observer_faithfulness:.3f} | {r.vs_observer_diff:+.3f} {r.vs_observer_ci} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    if not args.no_plots:
        warmups = set()
        for run in args.runs:
            cfg = run / "train_config.json"
            if cfg.exists():
                warmups.add(json.loads(cfg.read_text()).get("warmup"))
        plot_trajectories(summ, args.out / "trajectories.png",
                          warmup=warmups.pop() if len(warmups) == 1 else None)
        plot_distribution(df, args.out / "faithfulness_dist.png")
        plot_observer(df, args.out / "self_vs_observer.png")
        print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
