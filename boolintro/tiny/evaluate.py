"""Decisions, self-reports and their scores for the from-scratch model.

Scoring is the same as the main experiment: faithfulness asks whether the reported
rule picks the option the model itself picks on held-out pairs.  Reports are decoded
over rule tokens only (format-constrained), so an unparsable report is a malformed
formula, not chatter.  An attribute statement ("NOT x3") is scored as the one-literal
rule it states.

Per report kind (formula / attribute) the metrics carry that prefix:
    <kind>_faith, _faith_shuffled, _parse_rate, _exact, _tie, _decisive_acc,
    _sign_ok, _rel_mentioned, _irr_mentioned, _val_corr_avg, _greedy_faith, _greedy_exact
plus decision_acc, assoc_ceiling (best one-literal faithfulness) and degree1_mass.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .. import boolfn, metrics, rules
from . import data as D

STATS = ("tie", "decisive_acc", "sign_ok", "rel_mentioned", "irr_mentioned")


def _last_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(input_ids=x, attention_mask=(x != D.PAD).long(), logits_to_keep=1).logits[:, -1].float()


@torch.no_grad()
def decision_p_first(model, spec, cids, device, batch: int = 8192) -> dict[int, list[float]]:
    """Position-debiased P(prefer first object) on each character's eval pairs."""
    chars = spec["characters"]
    rows = [(c, a, b) for c in cids for a, b in chars[c]["eval_pairs"]]
    if not rows:
        return {}
    c, a, b = map(np.array, zip(*rows))
    out = []
    for x in (D.decision_prompts(c, a, b), D.decision_prompts(c, b, a)):
        ps = []
        for i in range(0, len(x), batch):
            lg = _last_logits(model, torch.from_numpy(x[i:i + batch]).to(device))
            ps.append(torch.softmax(lg[:, [D.TOK["A"], D.TOK["B"]]], -1)[:, 0].cpu())
        out.append(torch.cat(ps).numpy())
    p_first = metrics.debiased_first(out[0], out[1])
    res, i = {}, 0
    for cid in cids:
        n = len(chars[cid]["eval_pairs"])
        res[cid] = [float(p) for p in p_first[i:i + n]]
        i += n
    return res


@torch.no_grad()
def generate_reports(model, cids, device, n_samples: int, temperature: float, seed: int,
                     kind: str = "formula", batch: int = 4096) -> dict[int, tuple[list[int], list[list[int]]]]:
    """Per character: (greedy report tokens, [sampled report tokens])."""
    g = torch.Generator(device=device).manual_seed(seed)
    allowed = torch.full((model.config.vocab_size,), float("-inf"), device=device)
    allowed[[D.TOK[t] for t in D.RULE_TOKENS] + [D.EOS]] = 0.0
    per = 1 + n_samples
    owners = np.repeat(np.asarray(cids, dtype=np.int64), per)
    greedy_row = np.tile(np.arange(per) == 0, len(cids))
    seqs = []
    for i in range(0, len(owners), batch):
        x = torch.from_numpy(D.report_prompts(owners[i:i + batch], kind)).to(device)
        greedy = torch.from_numpy(greedy_row[i:i + batch]).to(device)
        done = torch.zeros(len(x), dtype=torch.bool, device=device)
        for _ in range(D.MAX_RULE_LEN):
            lg = _last_logits(model, x) + allowed
            samp = torch.multinomial(torch.softmax(lg / max(temperature, 1e-6), -1), 1, generator=g)[:, 0]
            nxt = torch.where(greedy, lg.argmax(-1), samp)
            nxt = torch.where(done, torch.full_like(nxt, D.PAD), nxt)
            x = torch.cat([x, nxt[:, None]], 1)
            done |= nxt == D.EOS
            if done.all():
                break
        seqs.extend(x[:, D.REPORT_PROMPT_LEN:].cpu().tolist())
    return {cid: (seqs[j * per], seqs[j * per + 1:(j + 1) * per]) for j, cid in enumerate(cids)}


def _nanmean(xs):
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


def _score_kind(kind, c, pairs, beh, target, greedy_toks, sample_toks) -> tuple[dict, list]:
    p = f"{kind}_"
    parsed = [D.parse_tokens(t) for t in sample_toks]
    ok = [r for r in parsed if r is not None]
    sts = [boolfn.report_stats(r, target, pairs, beh, D.K) for r in ok]
    m = {p + "parse_rate": len(ok) / len(parsed) if parsed else math.nan,
         p + "faith": metrics.mean_over_reports(ok, pairs, beh),
         p + "exact": _nanmean([float(rules.truth_table(r, D.K) == c["truth_table"]) for r in ok])}
    for s in STATS:
        m[p + s] = _nanmean([st[s] for st in sts])
    vt = boolfn.valence(target, D.K)
    m[p + "val_corr_avg"] = (boolfn.corr(np.mean([boolfn.valence(r, D.K) for r in ok], 0), vt)
                             if ok else math.nan)
    g = D.parse_tokens(greedy_toks)
    m[p + "greedy"] = D.decode_rule(greedy_toks)
    m[p + "greedy_faith"] = metrics.rule_score(g, pairs, beh) if g is not None else math.nan
    m[p + "greedy_exact"] = float(rules.truth_table(g, D.K) == c["truth_table"]) if g is not None else math.nan
    return m, ok


def evaluate(model, spec, groups, device, n_samples: int = 8, temperature: float = 1.0,
             seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Returns (per-character rows, per (group, family) summary rows)."""
    model.eval()
    chars = spec["characters"]
    cids = D.ids_of(spec, groups)
    kinds = D.report_kinds(spec)
    p_first = decision_p_first(model, spec, cids, device)
    reps = {k: generate_reports(model, cids, device, n_samples, temperature, seed + i, kind=k)
            for i, k in enumerate(kinds)}
    model.train()

    per, parsed_of, beh_of = [], defaultdict(dict), {}
    for cid in cids:
        c = chars[cid]
        target = D.char_rule(c)
        pairs = [tuple(p) for p in c["eval_pairs"]]
        beh = [a if p > 0.5 else b for (a, b), p in zip(pairs, p_first[cid])]
        tgt = [a if rules.evaluate(target, a) else b for a, b in pairs]
        m = {"id": cid, "group": c["group"], "level": c["level"], "rule": c["rule"],
             "decision_acc": float(np.mean([x == t for x, t in zip(beh, tgt)])),
             "assoc_ceiling": boolfn.association_ceiling(target, pairs, beh, D.K),
             "degree1_mass": float(boolfn.degree_mass(target, D.K)[0])}
        for k in kinds:
            mk, ok = _score_kind(k, c, pairs, beh, target, *reps[k][cid])
            m.update(mk)
            parsed_of[k][cid] = ok
        per.append(m)
        beh_of[cid] = (pairs, beh)

    # Shuffled-character baseline: each character's reports scored against another
    # character's behaviour (same group and family).
    rng = np.random.default_rng(seed)
    by_gl = defaultdict(list)
    for m in per:
        by_gl[(m["group"], str(m["level"]))].append(m)
    for ms in by_gl.values():
        order = rng.permutation(len(ms))
        for k in kinds:
            for pos, i in enumerate(order):
                j = order[(pos + 1) % len(order)]
                pairs, beh = beh_of[ms[i]["id"]]
                ms[i][f"{k}_faith_shuffled"] = (math.nan if i == j else
                                                metrics.mean_over_reports(parsed_of[k][ms[j]["id"]], pairs, beh))

    summ = []
    num = [key for key in per[0] if key not in ("id", "group", "level", "rule") and not key.endswith("_greedy")]
    for (grp, lv), ms in sorted(by_gl.items()):
        row = {"group": grp, "level": ms[0]["level"], "n": len(ms)}
        for key in num:
            row[key] = _nanmean([m[key] for m in ms])
        summ.append(row)
    return per, summ
