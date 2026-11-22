"""Aggregate evaluation results across runs and checkpoints; write tables and figures.

    python -m boolintro.analyze --runs runs/L1 runs/L2 runs/L3 runs/L4 --out results/main

Outputs:
  per_character.csv     one row per (run, step, character)
  summary.csv           mean and SEM over characters per (run, level, step)
  report.md             final-checkpoint table with bootstrap CIs for the key contrasts
  trajectories.png      decision accuracy and faithfulness over training, per level
  trajectories_breaks.png  the same, marking where self-reports break (parse rate < 25%)
  parse_rate.png        share of self-reports that parse as a formula, per level
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
    if not ys:
        return []
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


def plot_trajectories_parse(summ: pd.DataFrame, path: Path, warmup: int | None = None,
                            parse_floor: float = 0.25, parse_path: Path | None = None):
    """Trajectories marking where self-reports break, plus the parse rate as a separate image.

    A level counts as broken from the first checkpoint where its mean parse rate falls below
    `parse_floor`; faithfulness after that rests on very few reports and is drawn faded.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    groups = [(lv, s.sort_values("step").reset_index(drop=True)) for (_, lv), s in summ.groupby(["run", "level"])]
    last_step = int(summ.step.max())
    breaks = {}
    for lv, s in groups:
        low = s[s.parse_rate_mean < parse_floor]
        breaks[lv] = int(low.step.iloc[0]) if len(low) else None

    def draw(ax, key, title, sub):
        _style(ax)
        lo = 0.0 if key == "parse_rate" else 0.32
        if key != "parse_rate":
            ax.hlines(0.5, -last_step * 0.02, last_step, color=MUTED, lw=1, ls=":")
            ax.text(last_step * 0.01, 0.492, "chance", color=MUTED, fontsize=8, va="top")
        else:
            ax.hlines(parse_floor, -last_step * 0.02, last_step, color=MUTED, lw=1, ls="--")
            ax.text(last_step * 0.99, parse_floor + 0.015, f"break point: {parse_floor:.0%} parse",
                    color=MUTED, fontsize=8, ha="right", va="bottom")
        if warmup:
            ax.axvline(warmup, color=MUTED, lw=1, ls=(0, (2, 3)))
            ax.text(warmup + 80, lo + 0.015, "LR warmup ends", color=MUTED, fontsize=8, va="bottom")
        ends = []
        for lv, s in groups:
            col = LEVEL_COLORS.get(lv, INK)
            b = breaks[lv]
            ok = s if b is None or key != "faithfulness" else s[s.step <= b]
            bad = s.iloc[0:0] if b is None or key != "faithfulness" else s[s.step >= b]
            m, e = ok[f"{key}_mean"], ok[f"{key}_sem"].fillna(0)
            if key == "faithfulness":
                okb = ok[ok.step < b] if b is not None else ok
                ax.plot(okb.step, okb["shuffled_faithfulness_mean"], color=col, lw=1, ls="--", alpha=0.7)
            ax.fill_between(ok.step, m - e, m + e, color=col, alpha=0.15, lw=0)
            ax.plot(ok.step, m, color=col, lw=2, marker="o", ms=4, markeredgecolor="white", markeredgewidth=0.8)
            if len(bad):
                ax.plot(bad.step, bad[f"{key}_mean"], color=col, lw=1.2, ls=(0, (3, 2)), alpha=0.45,
                        marker="o", ms=4, markerfacecolor="white", markeredgecolor=col)
            if b is not None:
                ax.axvline(b, color=col, lw=1.2, ls=":", alpha=0.8)
            valid = ok.dropna(subset=[f"{key}_mean"])
            if len(valid):
                ends.append((lv, float(valid[f"{key}_mean"].iloc[-1]), int(valid.step.iloc[-1]), col,
                             key == "faithfulness" and b is not None))
        if key == "faithfulness":
            # Break notes next to each break point: the highest one above its point, the rest below.
            marks = sorted([(lv, breaks[lv], float(s.loc[s.step == breaks[lv], "faithfulness_mean"].iloc[0]))
                            for lv, s in groups if breaks[lv] is not None], key=lambda t: -t[2])
            for i, (lv, b, yb) in enumerate(marks):
                ty = yb + 0.15 if i == 0 else yb - 0.07 - 0.07 * (i - 1)
                ax.annotate(f"L{lv} reports break at step {b}\nlast reliable value {yb:.2f}", xy=(b, yb),
                            xytext=(b + last_step * 0.03, ty), color=INK, fontsize=8.5, va="center",
                            arrowprops=dict(arrowstyle="-", color=LEVEL_COLORS.get(lv, INK), lw=1,
                                            shrinkA=2, shrinkB=4))

        def series(lv_skip=None):
            return [(lv2, s2.dropna(subset=[f"{key}_mean"])) for lv2, s2 in groups if lv2 != lv_skip]

        # Runs that reach the last step: labels in the right margin (dodged, kept inside the axes).
        # Broken runs get no margin label (their break note says it).  Runs that stop early: label
        # next to their last point, on whichever side and height has the most clearance.
        full = [e for e in ends if e[2] == last_step and not e[4]]
        early = [e for e in ends if e[2] != last_step and not e[4]]
        label_y = _dodge([max(y, lo + 0.03) + 1e-3 * y for _, y, _, _, _ in full], gap=0.05, top=1.02)
        for (lv, y, st, col, _), ly in zip(full, label_y):
            ax.annotate(f"L{lv} {y:.2f}", xy=(st, y), xytext=(last_step * 1.04, ly), color=INK,
                        fontsize=9, va="center", arrowprops=dict(arrowstyle="-", color=col, lw=1, shrinkA=0, shrinkB=3))
        placed = []
        span, gapx = last_step * 0.14, last_step * 0.03
        for lv, y, st, col, _ in sorted(early, key=lambda e: -e[1]):
            best, best_score = (st + gapx, "left", y), -1e9
            for side in ("right", "left"):
                xt = st + gapx if side == "right" else st - gapx
                xs = np.linspace(xt, xt + span, 12) if side == "right" else np.linspace(xt - span, xt, 12)
                others = [np.interp(xs, v.step, v[f"{key}_mean"], left=np.nan, right=np.nan)
                          for _, v in series(lv if side == "right" else None)]
                for cand in np.clip(y + np.linspace(-0.15, 0.15, 31), lo + 0.03, 1.0):
                    if any(abs(cand - p) < 0.05 for p in placed):
                        continue
                    clear = min([np.nanmin(np.abs(o - cand)) for o in others if np.isfinite(o).any()] + [1.0])
                    score = min(clear, 0.08) - 0.2 * abs(cand - y) - (0.01 if side == "left" else 0)
                    if score > best_score:
                        best, best_score = (xt, "left" if side == "right" else "right", cand), score
            xt, ha, ly = best
            placed.append(ly)
            ax.annotate(f"L{lv} {y:.2f}", xy=(st, y), xytext=(xt, ly), color=INK, fontsize=9, va="center", ha=ha,
                        arrowprops=dict(arrowstyle="-", color=col, lw=1, shrinkA=0, shrinkB=3))
        ax.set_xlim(-last_step * 0.02, last_step * 1.27)
        ax.set_xticks(range(0, last_step + 1, 2000))
        ax.set_ylim(lo, 1.04)
        ax.set_title(title, color=INK, fontsize=12, loc="left", pad=20, fontweight="bold")
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")
        ax.set_xlabel("Training step", color=MUTED)
    level_handles = [Line2D([], [], color=LEVEL_COLORS.get(lv, INK), lw=2, marker="o", ms=4,
                            label=LEVEL_NAMES.get(lv, f"L{lv}")) for lv, _ in groups]

    # Figure 1: decision accuracy and faithfulness, with the break points marked.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    draw(axes[0], "decision_acc", "A   Decision accuracy", "model's choice matches the training target")
    draw(axes[1], "faithfulness", "B   Self-report faithfulness", "reported rule predicts the model's own choice")
    axes[0].set_ylabel("Mean over characters (chance = 0.5)", color=MUTED)
    handles = level_handles + [
        Line2D([], [], color=MUTED, lw=1, ls="--", label="shuffled baseline (B)"),
        Line2D([], [], color=MUTED, lw=1.2, ls=(0, (3, 2)), marker="o", ms=4, markerfacecolor="white",
               label="after reports break (unreliable)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               handlelength=2.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=2.5)
    fig.savefig(path, dpi=200)
    plt.close(fig)

    # Figure 2 (separate image): the parse rate that defines the break points.
    fig, ax = plt.subplots(figsize=(7, 4.8))
    draw(ax, "parse_rate", "Reports that parse as a rule", "share of self-reports that are a valid formula")
    ax.set_ylabel("Share of reports", color=MUTED)
    fig.legend(handles=level_handles, loc="lower center", ncol=2, frameon=False, fontsize=9, handlelength=2.4)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(parse_path or path.with_name(path.stem + "_parse_rate.png"), dpi=200)
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
        wu = warmups.pop() if len(warmups) == 1 else None
        plot_trajectories(summ, args.out / "trajectories.png", warmup=wu)
        plot_trajectories_parse(summ, args.out / "trajectories_breaks.png", warmup=wu,
                                parse_path=args.out / "parse_rate.png")
        plot_distribution(df, args.out / "faithfulness_dist.png")
        plot_observer(df, args.out / "self_vs_observer.png")
        print(f"figures written to {args.out}")


if __name__ == "__main__":
    main()
