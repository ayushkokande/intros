"""Cross-model control for the from-scratch GPT-2 (fast, CPU-friendly version of
boolintro.crossmodel; see that module for the idea).

Three GPT-2s per seed, all pretrained on the same data (choices + reports for the class-1
characters; only the new characters matter here):

    A    the self model: finetuned on the new characters' rule-labelled choices
    A2   same size as A, different initial weights and data order        -> outsider "same"
    B    a different size (default 2 layers x 256 wide vs 4 x 128)        -> outsider "other"

Outsiders are finetuned on A's recorded choices instead of the rules.  Role swap: B is also
finetuned on the rules as a self model, with A (finetuned on B's choices) as its outsider.
Reports from self and outsider use the same report prompt and are scored against the self
model's held-out choices.  There is no prose here, so no third-person prompt (that control
exists only in the Hugging Face version).

All in one go for one seed (roughly 30-60 minutes on a laptop CPU with the defaults):

    python -m boolintro.tiny.crossmodel run --exp runs/tiny_xm/s0 --seed 0
    python -m boolintro.crossmodel summarize --exps runs/tiny_xm/s*/self_* --out runs/tiny_xm/summary

Rerunning skips finished stages.  Pieces, for existing tiny runs:

    python -m boolintro.tiny.crossmodel record --m1-run R --ft ft_embed_s0 --exp E
    python -m boolintro.tiny.train finetune --run-dir R2 --name ft_from_R --labels-from E/m1_choices.json
    python -m boolintro.tiny.crossmodel evaluate --exp E --m2-run R2 --m2-ft ft_from_R --condition same
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .. import crossmodel as X
from .. import metrics
from . import data as D
from . import evaluate as E
from . import model as M


@torch.no_grad()
def p_first_on(model, spec, cids, pair_key, device, batch: int = 8192) -> dict[int, list[float]]:
    """Debiased P(prefer first object) on each character's `pair_key` pairs."""
    model.eval()
    chars = spec["characters"]
    rows = [(c, a, b) for c in cids for a, b in chars[c][pair_key]]
    c, a, b = map(np.array, zip(*rows))
    out = []
    for x in (D.decision_prompts(c, a, b), D.decision_prompts(c, b, a)):
        ps = []
        for i in range(0, len(x), batch):
            lg = E._last_logits(model, torch.from_numpy(x[i:i + batch]).to(device))
            ps.append(torch.softmax(lg[:, [D.TOK["A"], D.TOK["B"]]], -1)[:, 0].cpu())
        out.append(torch.cat(ps).numpy())
    pf = metrics.debiased_first(out[0], out[1])
    res, i = {}, 0
    for cid in cids:
        n = len(chars[cid][pair_key])
        res[cid] = [float(p) for p in pf[i:i + n]]
        i += n
    return res


def load_preferences(path: Path, spec: dict) -> dict[int, dict]:
    """{character id: {sorted pair: chosen object}} from m1_choices.json, checked against spec."""
    rec = json.loads(Path(path).read_text())
    out = {}
    for r in rec["characters"]:
        c = spec["characters"][r["id"]]
        if [list(p) for p in r["train_pairs"]] != c["train_pairs"] or r["rule"] != c["rule"]:
            raise SystemExit(f"{path} does not match this spec (character {r['id']}); "
                             "pretrain M2 with the same spec options and --seed as M1")
        out[r["id"]] = {tuple(p): ch for p, ch in zip(r["train_pairs"], r["train_choice"])}
    return out


def describe_model(run: Path) -> str:
    cfg = json.loads((Path(run) / "pretrain" / "config.json").read_text())
    ms = cfg.get("model_seed")
    ms = cfg["seed"] if ms is None else ms
    return f"gpt2-{cfg['n_layer']}x{cfg['n_embd']}-init{ms}"


def record(args):
    exp = args.exp
    exp.mkdir(parents=True, exist_ok=True)
    if (exp / "m1_choices.json").exists():
        print(f"{exp / 'm1_choices.json'} exists")
        return
    spec = D.load_spec(args.m1_run / "spec.json")
    dev = M.device_of(args.device)
    model = M.load(args.m1_run / args.ft / "model", dev)
    cids = D.ids_of(spec, ("new",))
    p_tr = p_first_on(model, spec, cids, "train_pairs", dev)
    p_ev = p_first_on(model, spec, cids, "eval_pairs", dev)
    rows = []
    for cid in cids:
        c = spec["characters"][cid]
        tt = c["truth_table"]
        tch = [X.choice(a, b, p) for (a, b), p in zip(c["train_pairs"], p_tr[cid])]
        rows.append({"id": cid, "level": c["level"], "rule": c["rule"], "train_pairs": c["train_pairs"],
                     "train_choice": tch,
                     "train_acc": float(np.mean([x == (a if (tt >> a) & 1 else b)
                                                 for (a, b), x in zip(c["train_pairs"], tch)])),
                     "eval_p_first": [round(p, 5) for p in p_ev[cid]]})
    D.save_spec(spec, exp / "spec.json")
    (exp / "m1_choices.json").write_text(json.dumps(
        {"m1_run": str(args.m1_run), "ft": args.ft, "model": describe_model(args.m1_run),
         "seed": spec["seed"], "characters": rows}, indent=1))
    print(f"recorded {describe_model(args.m1_run)} choices for {len(rows)} new characters (agreement "
          f"with the rule on train pairs {np.mean([r['train_acc'] for r in rows]):.3f})", flush=True)


def _with_attrs(chars):
    return [{**c, "attributes": D.ATTRS} for c in chars]


def evaluate(args):
    exp = args.exp
    rec = json.loads((exp / "m1_choices.json").read_text())
    spec = D.load_spec(exp / "spec.json")
    m2_spec = json.loads((args.m2_run / "spec.json").read_text())
    if m2_spec["characters"] != spec["characters"]:
        raise SystemExit(f"{args.m2_run} has a different spec from M1")
    dev = M.device_of(args.device)
    cids = [r["id"] for r in rec["characters"]]
    chars = _with_attrs([spec["characters"][c] for c in cids])
    m1_p = [r["eval_p_first"] for r in rec["characters"]]
    reports = {}
    m2_p = None
    for role, path in (("m1", Path(rec["m1_run"]) / rec["ft"] / "model"),
                       ("m2", args.m2_run / args.m2_ft / "model")):
        model = M.load(path, dev)
        reps = E.generate_reports(model, cids, dev, args.n_samples, args.temperature,
                                  seed=args.seed + (0 if role == "m1" else 104729))
        reports[f"{role}_self"] = [[D.parse_tokens(t) for t in reps[c][1]] for c in cids]
        if role == "m2":
            pe = p_first_on(model, spec, cids, "eval_pairs", dev)
            m2_p = [pe[c] for c in cids]
        del model
    per = X.score(chars, m1_p, m2_p, reports)
    out = {"setup": "tiny", "seed": spec["seed"], "condition": args.condition,
           "m1": f"{rec['m1_run']}/{rec['ft']}", "m2": f"{args.m2_run}/{args.m2_ft}",
           "m1_model": rec["model"].rsplit("-init", 1)[0], "m2_model": describe_model(args.m2_run).rsplit("-init", 1)[0],
           "n_reports": args.n_samples, "temperature": args.temperature, "per_character": per}
    (exp / f"crossmodel_eval_{args.condition}.json").write_text(json.dumps(out, indent=1))
    print(f"self {rec['model']} <- outsider {describe_model(args.m2_run)} ({args.condition})", flush=True)
    X.print_levels(per)


def run(args):
    """Pretrain A, A2, B; finetune selves on rules, outsiders on recorded choices; evaluate."""
    from . import train

    exp = args.exp
    spec_opts = ["--seed", str(args.seed), "--families", args.families, "--n-pre-report", str(args.n_pre_report),
                 "--n-pre-choice", str(args.n_pre_choice), "--n-new", str(args.n_new)]
    common = ["--device", args.device, "--batch-size", str(args.batch_size), "--n-samples", str(args.n_samples)]
    pre = ["--steps", str(args.pretrain_steps), "--eval-every", str(args.pretrain_steps), "--warmup",
           str(max(1, min(1000, args.pretrain_steps // 10)))]
    ft = ["--seed", str(args.seed), "--steps", str(args.finetune_steps), "--eval-every", str(args.finetune_steps)]
    size = lambda L, d, h: ["--n-layer", str(L), "--n-embd", str(d), "--n-head", str(h)]  # noqa: E731
    models = {"A": (size(args.n_layer, args.n_embd, args.n_head), args.seed),
              "A2": (size(args.n_layer, args.n_embd, args.n_head), args.seed + 1001),
              "B": (size(args.other_n_layer, args.other_n_embd, args.other_n_head), args.seed + 2002)}
    for name, (sz, mseed) in models.items():
        print(f"== pretrain {name}", flush=True)
        train.main(["pretrain", "--run-dir", str(exp / name), *spec_opts, *common, *pre, *sz,
                    "--model-seed", str(mseed)])

    def self_and_outsiders(self_name, outsiders):
        print(f"== self {self_name}: finetune on the rules", flush=True)
        train.main(["finetune", "--run-dir", str(exp / self_name), "--name", "ft_rule", *common, *ft])
        sexp = exp / f"self_{self_name}"
        record(argparse.Namespace(exp=sexp, m1_run=exp / self_name, ft="ft_rule", device=args.device))
        for cond, o in outsiders:
            print(f"== outsider {o} ({cond}) for {self_name}: finetune on {self_name}'s choices", flush=True)
            train.main(["finetune", "--run-dir", str(exp / o), "--name", f"ft_from_{self_name}",
                        "--labels-from", str(sexp / "m1_choices.json"), *common, *ft])
            evaluate(argparse.Namespace(exp=sexp, m2_run=exp / o, m2_ft=f"ft_from_{self_name}", condition=cond,
                                        device=args.device, n_samples=args.n_reports,
                                        temperature=args.temperature, seed=args.seed))

    self_and_outsiders("A", [("same", "A2"), ("other", "B")])
    if not args.no_swap:
        self_and_outsiders("B", [("other", "A")])


def get_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record")
    r.add_argument("--m1-run", type=Path, required=True)
    r.add_argument("--ft", default="ft_embed_s0", help="M1's finetune subdirectory")
    r.add_argument("--exp", type=Path, required=True)
    r.add_argument("--device", default="auto")
    e = sub.add_parser("evaluate")
    e.add_argument("--exp", type=Path, required=True)
    e.add_argument("--m2-run", type=Path, required=True)
    e.add_argument("--m2-ft", required=True, help="the outsider's finetune subdirectory")
    e.add_argument("--condition", required=True, help="e.g. same or other")
    e.add_argument("--n-samples", type=int, default=24)
    e.add_argument("--temperature", type=float, default=1.0)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--device", default="auto")
    a = sub.add_parser("run", help="the whole experiment for one seed")
    a.add_argument("--exp", type=Path, required=True)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--families", default="L1,L2,L3,L4")
    a.add_argument("--n-pre-report", type=int, default=400)
    a.add_argument("--n-pre-choice", type=int, default=0)
    a.add_argument("--n-new", type=int, default=80)
    a.add_argument("--pretrain-steps", type=int, default=5000)
    a.add_argument("--finetune-steps", type=int, default=1500)
    a.add_argument("--batch-size", type=int, default=256)
    a.add_argument("--n-layer", type=int, default=4)
    a.add_argument("--n-embd", type=int, default=128)
    a.add_argument("--n-head", type=int, default=4)
    a.add_argument("--other-n-layer", type=int, default=2, help="size of B, the other-family model")
    a.add_argument("--other-n-embd", type=int, default=256)
    a.add_argument("--other-n-head", type=int, default=8)
    a.add_argument("--no-swap", action="store_true", help="skip the role swap (B as self, A as outsider)")
    a.add_argument("--n-samples", type=int, default=4, help="reports per character in the training-time evals")
    a.add_argument("--n-reports", type=int, default=24, help="reports per character in the final comparison")
    a.add_argument("--temperature", type=float, default=1.0)
    a.add_argument("--device", default="auto")
    return p.parse_args(argv)


def main(argv=None):
    args = get_args(argv)
    {"record": record, "evaluate": evaluate, "run": run}[args.cmd](args)


if __name__ == "__main__":
    main()
