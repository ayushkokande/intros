"""Linear probes: what does a character's learned representation store?

    python -m boolintro.tiny.probe --run-dir runs/tiny/formula_s0

For each character we read its representation from the model:
    embed     the input-embedding row of its token
    layerN    the residual stream at its token after block N, for the input "<bos> <cN>"
              (causal attention, so this depends on the character alone)
and fit ridge probes for its rule's Fourier coefficients (boolintro.boolfn):
    deg1      the 5 single-attribute weights: the "preference vector" P
    deg2      the 10 pairwise interactions
    deg3      the 10 three-way interactions (all parity information lives here)
    truth     the full truth table (32 values in +-1)

Probes are fit on the pretraining characters (pre_report + pre_choice) and scored
    pre_report, pre_choice   5-fold cross-validated
    new                      transfer: the probe fit on pretraining characters, applied
                             to the new characters' finetuned representations
    new_untrained            the same, on the new characters before finetuning (baseline)

Metrics: r2 (pooled over characters and coefficients), and per family the mean cosine
between predicted and true coefficient vectors (characters whose true vector is zero
are skipped), and truth-table sign accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .. import boolfn
from . import data as D
from . import model as M

TARGETS = ("deg1", "deg2", "deg3", "truth")
ALPHAS = (1e-2, 1e-1, 1, 10, 100, 1000)


def targets(spec) -> dict[str, np.ndarray]:
    rs = [D.char_rule(c) for c in spec["characters"]]
    out = {f"deg{d}": np.array([boolfn.degree_vector(r, d, D.K) for r in rs]) for d in (1, 2, 3)}
    out["truth"] = np.array([[1.0 if (c["truth_table"] >> o) & 1 else -1.0 for o in range(2**D.K)]
                             for c in spec["characters"]])
    return out


@torch.no_grad()
def features(model, spec, device) -> dict[str, np.ndarray]:
    model.eval()
    n = len(spec["characters"])
    ids = torch.tensor([D.char_token(c) for c in range(n)], device=device)
    feats = {"embed": model.transformer.wte.weight[ids].float().cpu().numpy()}
    x = torch.stack([torch.full_like(ids, D.BOS), ids], 1)
    hs = model(input_ids=x, output_hidden_states=True).hidden_states
    for i, h in enumerate(hs[1:], 1):
        feats[f"layer{i}"] = h[:, 1].float().cpu().numpy()
    return feats


def _fit(X, Y, alpha):
    mx, my = X.mean(0), Y.mean(0)
    sx = X.std(0) + 1e-8
    Xs = (X - mx) / sx
    W = np.linalg.solve(Xs.T @ Xs + alpha * np.eye(X.shape[1]), Xs.T @ (Y - my))
    return lambda Z: ((Z - mx) / sx) @ W + my


def _r2(Y, P, ref_mean):
    sst = ((Y - ref_mean) ** 2).sum()
    return float(1 - ((Y - P) ** 2).sum() / sst) if sst > 0 else math.nan


def _folds(n, k, seed):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(k)]


def fit_probe(X, Y, seed=0):
    """Ridge with alpha chosen by 5-fold CV; returns (predict, cv_predictions)."""
    folds = _folds(len(X), 5, seed)
    best, best_cv = None, None
    for a in ALPHAS:
        cv = np.zeros_like(Y)
        for f in folds:
            tr = np.setdiff1d(np.arange(len(X)), f)
            cv[f] = _fit(X[tr], Y[tr], a)(X[f])
        score = _r2(Y, cv, Y.mean(0))
        if best is None or score > best[0]:
            best, best_cv = (score, a), cv
    return _fit(X, Y, best[1]), best_cv


def _cos(Y, P):
    out = []
    for y, p in zip(Y, P):
        ny, npred = np.linalg.norm(y), np.linalg.norm(p)
        if ny > 1e-9 and npred > 1e-12:
            out.append(float(y @ p / (ny * npred)))
    return float(np.mean(out)) if out else math.nan


def probe_model(spec, feats, feats_untrained, tg, model_name) -> list[dict]:
    chars = spec["characters"]
    group = np.array([c["group"] for c in chars])
    fam = np.array([str(c["level"]) for c in chars])
    pre = np.isin(group, ["pre_report", "pre_choice"])
    rows = []
    for rep, X in feats.items():
        for tname, Y in tg.items():
            predict, cv = fit_probe(X[pre], Y[pre])
            ref = Y[pre].mean(0)
            preds = {"pre_report": (cv[group[pre] == "pre_report"], group == "pre_report"),
                     "pre_choice": (cv[group[pre] == "pre_choice"], group == "pre_choice")}
            new = group == "new"
            if new.any():
                if feats_untrained is None:  # the pretrained model: new rows are still random
                    preds["new_untrained"] = (predict(X[new]), new)
                else:
                    preds["new"] = (predict(X[new]), new)
                    preds["new_untrained"] = (predict(feats_untrained[rep][new]), new)
            for gname, (P, mask) in preds.items():
                if not mask.any():
                    continue
                Yg = Y[mask]
                base = {"model": model_name, "rep": rep, "target": tname, "group": gname}
                rows.append({**base, "family": "all", "metric": "r2", "value": _r2(Yg, P, ref)})
                for f in sorted(set(fam[mask])):
                    fm = fam[mask] == f
                    rows.append({**base, "family": f, "metric": "cos", "value": _cos(Yg[fm], P[fm])})
                    if tname == "truth":
                        rows.append({**base, "family": f, "metric": "sign_acc",
                                     "value": float((np.sign(P[fm]) == Yg[fm]).mean())})
    return rows


def plot(df: pd.DataFrame, path: Path, model_name: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..analyze import INK, MUTED, _style

    colors = {"deg1": "#2a78d6", "deg2": "#eb6834", "deg3": "#1baf7a", "truth": "#6b6a64"}
    labels = {"deg1": "degree 1 (P)", "deg2": "degree 2", "deg3": "degree 3", "truth": "full truth table"}
    d = df[(df.model == model_name) & (df.metric == "r2")]
    groups = [g for g in ("pre_report", "pre_choice", "new", "new_untrained") if g in set(d.group)]
    reps = list(dict.fromkeys(d.rep))
    fig, axes = plt.subplots(1, len(groups), figsize=(4.2 * len(groups), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, g in zip(axes, groups):
        _style(ax)
        for t in TARGETS:
            s = d[(d.group == g) & (d.target == t)].set_index("rep").reindex(reps)
            ax.plot(range(len(reps)), s.value.clip(lower=-0.2), color=colors[t], lw=2, marker="o", ms=4,
                    markeredgecolor="white", label=labels[t])
        ax.axhline(0, color=MUTED, lw=1, ls=":")
        ax.set_xticks(range(len(reps)), [r.replace("layer", "L") for r in reps], fontsize=8)
        ax.set_title(g.replace("_", " "), color=INK, fontsize=11, loc="left", fontweight="bold")
        ax.set_xlabel("representation (embedding, then after each block)", color=MUTED, fontsize=8.5)
    axes[0].set_ylabel("probe R² (linear)", color=MUTED)
    axes[-1].legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.suptitle(f"What the character representation linearly encodes  ({model_name})", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--models", nargs="*", default=None,
                   help="subdirs with a saved model (default: pretrain and every finished finetune)")
    p.add_argument("--device", default="auto")
    args = p.parse_args(argv)
    run = args.run_dir
    spec = D.load_spec(run / "spec.json")
    dev = M.device_of(args.device)
    names = args.models or ["pretrain"] + sorted(d.name for d in run.iterdir()
                                                 if d.name.startswith("ft_") and (d / "DONE").exists())
    tg = targets(spec)
    untrained = features(M.load(run / "pretrain" / "model", dev), spec, dev)  # new rows never trained
    out = run / "probe"
    out.mkdir(exist_ok=True)
    rows = []
    for name in names:
        feats = untrained if name == "pretrain" else features(M.load(run / name / "model", dev), spec, dev)
        rows += probe_model(spec, feats, None if name == "pretrain" else untrained, tg, name)
        print(f"probed {name}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "probe.csv", index=False)
    for name in names:
        plot(df, out / f"probe_{name}.png", name)
    best = (df[(df.metric == "r2") & (df.family == "all")]
            .groupby(["model", "group", "target"]).value.max().unstack("target"))
    (out / "probe_summary.md").write_text("Best layer's probe R²\n\n" + best.round(2).to_string() + "\n")
    print(best.round(2).to_string())


if __name__ == "__main__":
    main()
