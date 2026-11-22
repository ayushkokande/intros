"""Why do regex self-reports fail (or look faithful) at some levels?  Diagnosis of saved reports.

    python -m regexintro.diagnose --runs runs/regex-qwen3-32b_L1 runs/regex-qwen3-32b_L2 \
        runs/regex-qwen3-32b_L3 runs/regex-qwen3-32b_L4 --out runs/regex-qwen3-32b_diagnosis

Regex counterpart of boolintro.diagnose.  Needs only the eval outputs (no GPU).  For every
checkpoint, character and parsed report:

  literal         the report is a bare string with no regex operators (accepts one string)
  accept_all      the report accepts every string (e.g. ".*")
  accept_none     the report accepts no string of the universe
  in_family       the report defines exactly the language of *some* level template, with any letters
  right_level     ... a template of the character's own level
  same_template   ... the character's own template, possibly with the wrong letters
  exact           ... the character's own grammar (= exact_match in evaluate.py)

and the shuffled baseline for error-string faithfulness (missing from evaluate.py):

  error_faith           agreement of the character's reports with the model on the strings where
                        the model departs from the hidden grammar (as in metrics.py)
  shuffled_error_faith  the same strings and decisions, scored with the reports of the other
                        characters at the level whose grammar differs.  When the model is
                        one-sided (accepts almost nothing), restrictive reports agree with its
                        errors whoever they are about; error_faith - shuffled_error_faith
                        removes that.

Unparsed reports are sorted into: anchors (would parse with ^ / $ removed everywhere),
decision (a bare yes / no / accept / reject), and other.  The same statistics are computed
for the observer's reports (prefix obs_).

Outputs: per_character.csv, summary.csv, report.md, diagnosis.png.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from boolintro.analyze import INK, LEVEL_COLORS, MUTED, _style

from . import data, families, grammars
from .analyze import LEVEL_NAMES
from .metrics import bootstrap_ci

KINDS = ["literal", "accept_all", "accept_none", "in_family", "right_level", "same_template", "exact"]
_DECISION = re.compile(r"^\W*(yes|no|accept(ed)?|reject(ed)?)\b[\W\w]{0,40}$", re.I)


@functools.lru_cache(maxsize=1)
def universe_index() -> dict[str, int]:
    return {s: i for i, s in enumerate(grammars.universe())}


def report_kinds(rx: str, char: dict) -> dict:
    lang = grammars.language(rx)
    fam = families.identify_language(lang)
    level, ti, _ = families.template_of(char)
    return {"literal": float(set(rx) <= set(grammars.ALPHABET)),
            "accept_all": float(lang.all()), "accept_none": float(not lang.any()),
            "in_family": float(fam is not None),
            "right_level": float(fam is not None and fam[0] == level),
            "same_template": float(fam is not None and fam[:2] == (level, ti)),
            "exact": float(np.array_equal(lang, grammars.language(char["rule"])))}


def unparsed_kind(text: str) -> str:
    if _DECISION.match(text.strip()):
        return "decision"
    for line in text.splitlines()[:4]:
        if grammars.parse_report(line.replace("^", "").replace("$", "")) is not None:
            return "anchors"
    return "other"


def agreement_on(langs: list[np.ndarray], idx: np.ndarray, decisions: np.ndarray) -> float:
    """Mean over reports of the share of strings (universe indices) where the report agrees."""
    if not langs or not len(idx):
        return math.nan
    return float(np.mean([np.mean(lang[idx] == decisions) for lang in langs]))


def report_block(texts, char, prefix=""):
    parsed = [grammars.parse_report(t) for t in texts] if texts else []
    ok = [r for r in parsed if r is not None]
    row = {f"{prefix}n_reports": len(parsed),
           f"{prefix}parse_rate": len(ok) / len(parsed) if parsed else math.nan}
    kinds = [report_kinds(r, char) for r in ok]
    for k in KINDS:
        row[f"{prefix}{k}"] = float(np.mean([d[k] for d in kinds])) if kinds else math.nan
    bad = [unparsed_kind(t) for t, r in zip(texts or [], parsed) if r is None]
    for k in ("anchors", "decision", "other"):
        row[f"{prefix}unparsed_{k}"] = (bad.count(k) / len(parsed)) if parsed else math.nan
    return row, ok


def one_step(job) -> list[dict]:
    run, step_file = job
    spec = data.load_spec(run / "spec.json")
    by_id = {c["id"]: c for c in spec["characters"]}
    ev = json.loads(step_file.read_text())
    rep_file = step_file.with_name(step_file.stem + "_reports.jsonl")
    reps = {}
    if rep_file.exists():
        reps = {r["id"]: r for r in map(json.loads, rep_file.read_text().splitlines()) if r}
    uidx = universe_index()

    per = []  # (char, idx, behaviour, error mask, parsed report languages, row)
    for m in ev["per_character"]:
        c = by_id[m["id"]]
        idx = np.array([uidx[s] for s in c["eval_strings"]])
        beh = np.asarray(m["p_first"]) > 0.5
        tgt = grammars.language(c["rule"])[idx]
        r = reps.get(c["id"], {})
        row, ok = report_block(r.get("reports"), c)
        obs_row, _ = report_block(r.get("observer_reports"), c, prefix="obs_")
        langs = [grammars.language(x) for x in ok]
        err = beh != tgt
        row.update({"run": run.name, "step": ev["step"], "level": c["level"], "id": c["id"],
                    "name": c["name"], "rule": c["rule"],
                    "template": families.template_name(*families.template_of(c)[:2]),
                    "decision_acc": float(np.mean(beh == tgt)), "accept_rate": float(beh.mean()),
                    "faith": agreement_on(langs, idx, beh), "n_error": int(err.sum()),
                    "error_faith": agreement_on(langs, idx[err], beh[err])})
        if r.get("observer_reports"):
            row.update(obs_row)
        per.append((c, idx, beh, err, langs, row))

    for i, (c, idx, beh, err, _, row) in enumerate(per):
        peers = [p for j, p in enumerate(per) if j != i and p[0]["level"] == c["level"]
                 and p[0]["signature"] != c["signature"]]
        vals = [agreement_on(p[4], idx[err], beh[err]) for p in peers]
        vals = [v for v in vals if not math.isnan(v)]
        row["shuffled_error_faith"] = float(np.mean(vals)) if vals else math.nan
        row["shuffled_faith"] = float(np.nanmean([agreement_on(p[4], idx, beh) for p in peers])) \
            if peers and any(p[4] for p in peers) else math.nan
    return [p[5] for p in per]


def load(runs: list[Path], workers: int) -> pd.DataFrame:
    jobs = [(run, f) for run in runs for f in sorted((run / "eval").glob("step_*.json"))]
    if not jobs:
        raise SystemExit("no eval/step_*.json files found")
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            parts = list(ex.map(one_step, jobs))
    else:
        parts = [one_step(j) for j in jobs]
    return pd.DataFrame([r for p in parts for r in p])


STAT_COLS = (["parse_rate", *KINDS, "unparsed_anchors", "unparsed_decision", "unparsed_other",
              "decision_acc", "accept_rate", "faith", "shuffled_faith", "error_faith", "shuffled_error_faith"])


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in STAT_COLS + [f"obs_{k}" for k in ["parse_rate", *KINDS]] if c in df]
    g = df.groupby(["run", "level", "step"])
    return pd.concat([g[cols].mean(), g.size().rename("n_characters")], axis=1).reset_index()


def _ci(d: pd.Series) -> str:
    d = d.dropna()
    if not len(d):
        return "n/a"
    lo, hi = bootstrap_ci(d.tolist())
    return f"{d.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] (n={len(d)})"


def write_report(df: pd.DataFrame, path: Path, ref_step: int | None):
    lines = ["# Regex report diagnosis", "",
             "Shares are over parsed reports, then averaged over characters. Reference step: "
             + ("the final step of each run." if ref_step is None else f"{ref_step}."), ""]
    final = []
    for run, sub in df.groupby("run"):
        step = sub.step.max() if ref_step is None else ref_step
        final.append(sub[sub.step == step])
    fin = pd.concat(final) if final else df.iloc[:0]
    lines += ["## What the reports are", "",
              "| run | level | step | parse | literal | accept all | accept none | in family | right level "
              "| same template | exact | unparsed: anchors / decision / other |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (run, level, step), s in fin.groupby(["run", "level", "step"]):
        lines.append(f"| {run} | {level} | {step} | {s.parse_rate.mean():.2f} | {s.literal.mean():.2f} | "
                     f"{s.accept_all.mean():.2f} | {s.accept_none.mean():.2f} | {s.in_family.mean():.2f} | "
                     f"{s.right_level.mean():.2f} | {s.same_template.mean():.2f} | {s.exact.mean():.2f} | "
                     f"{s.unparsed_anchors.mean():.2f} / {s.unparsed_decision.mean():.2f} / "
                     f"{s.unparsed_other.mean():.2f} |")
    lines += ["", "## Error-string faithfulness against its shuffled baseline", "",
              "Paired over characters that make at least one error and have parsed reports. "
              "`error - shuffled` > 0 means the reports track this character's own errors beyond what "
              "any same-level report would.", "",
              "| run | level | step | accept rate | error faith | shuffled error faith | error − 0.5 | error − shuffled |",
              "|---|---|---|---|---|---|---|---|"]
    for (run, level, step), s in fin.groupby(["run", "level", "step"]):
        lines.append(f"| {run} | {level} | {step} | {s.accept_rate.mean():.2f} | {s.error_faith.mean():.3f} | "
                     f"{s.shuffled_error_faith.mean():.3f} | {_ci(s.error_faith - 0.5)} | "
                     f"{_ci(s.error_faith - s.shuffled_error_faith)} |")
    if "obs_parse_rate" in fin:
        lines += ["", "## Observer reports (same shares)", "",
                  "| run | level | step | parse | literal | in family | same template | exact |",
                  "|---|---|---|---|---|---|---|---|"]
        for (run, level, step), s in fin.dropna(subset=["obs_parse_rate"]).groupby(["run", "level", "step"]):
            lines.append(f"| {run} | {level} | {step} | {s.obs_parse_rate.mean():.2f} | "
                         f"{s.obs_literal.mean():.2f} | {s.obs_in_family.mean():.2f} | "
                         f"{s.obs_same_template.mean():.2f} | {s.obs_exact.mean():.2f} |")
    lines += ["", "## By template (reference step)", "",
              "| level | template | decision acc | faith | same template | exact | n |", "|---|---|---|---|---|---|---|"]
    for (level, template), s in fin.groupby(["level", "template"]):
        lines.append(f"| {level} | `{template}` | {s.decision_acc.mean():.3f} | {s.faith.mean():.3f} | "
                     f"{s.same_template.mean():.2f} | {s.exact.mean():.2f} | {len(s)} |")
    path.write_text("\n".join(lines) + "\n")


def plot(summ: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("same_template", "same template as the hidden grammar"), ("exact", "exact match"),
              ("literal", "bare-string reports"), ("error_faith", "error-string faithfulness"),
              ("shuffled_error_faith", "shuffled error-string faithfulness"), ("parse_rate", "parse rate")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True)
    for ax, (col, title) in zip(axes.flat, panels):
        for level, s in summ.groupby("level"):
            s = s.sort_values("step")
            ax.plot(s.step, s[col], color=LEVEL_COLORS.get(level, INK), lw=1.8,
                    label=LEVEL_NAMES.get(level, f"L{level}"))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        if col in ("error_faith", "shuffled_error_faith"):
            ax.axhline(0.5, color=MUTED, lw=0.8, ls="--")
        _style(ax)
    for ax in axes[-1]:
        ax.set_xlabel("training step", color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--ref-step", type=int, default=None, help="default: each run's final step")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.runs, args.workers)
    df.to_csv(args.out / "per_character.csv", index=False)
    summ = summarize(df)
    summ.to_csv(args.out / "summary.csv", index=False)
    write_report(df, args.out / "report.md", args.ref_step)
    if not args.no_plots:
        plot(summ, args.out / "diagnosis.png")
    print((args.out / "report.md").read_text())


if __name__ == "__main__":
    main()
