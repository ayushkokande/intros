"""Decisions, regex self-reports and their scores for the from-scratch model.

Port of boolintro.tiny.evaluate.  The metrics have the same names and definitions as the main
regex experiment (regexintro.metrics.character_metrics, regexintro.evaluate):

    decision_acc, decision_p_target, model_accept_rate, parse_rate,
    faithfulness, faithfulness_balanced, report_target_agreement, exact_match,
    report_accept_rate, n_error_pairs, error_faithfulness, shuffled_faithfulness

plus greedy_faithfulness / greedy_exact for the greedy report.  `score_character` computes
them in one pass per report (each report's match vector is computed once, and full language
equivalence is only checked for reports that already agree with the target on every eval
string); tests check it equals regexintro.metrics.character_metrics.

Reports are decoded over report tokens only (letters and regex symbols, format-constrained),
so an unparsable report is a malformed regex, not chatter.

Shuffled baseline: a character's reports scored against the behaviour of up to `n_peers`
other characters of the same group and family whose language differs (the main experiment
uses all such peers; a sample keeps the many-character pretraining evals fast).
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import torch

from .. import grammars, metrics
from . import data as D


def _nanmean(xs):
    v = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return float(np.mean(v)) if v else math.nan


@torch.no_grad()
def decision_p_accept(model, spec, cids, device, batch: int = 8192) -> dict[int, list[float]]:
    """P(Y) / (P(Y) + P(N)) at the "->" position, for each character's eval strings."""
    chars = spec["characters"]
    owner = np.concatenate([[c] * len(chars[c]["eval_strings"]) for c in cids]) if cids else []
    if not len(owner):
        return {}
    uidx = np.concatenate([D.universe_index(chars[c]["eval_strings"]) for c in cids])
    x, arrow = D.decision_prompts(owner, uidx)
    ps = []
    for i in range(0, len(x), batch):
        xb = torch.from_numpy(x[i:i + batch]).to(device)
        h = model.transformer(input_ids=xb, attention_mask=(xb != D.PAD).long()).last_hidden_state
        pos = torch.from_numpy(arrow[i:i + batch]).to(device)
        lg = model.lm_head(h[torch.arange(len(xb), device=device), pos]).float()
        ps.append(torch.softmax(lg[:, [D.YES, D.NO]], -1)[:, 0].cpu())
    p = torch.cat(ps).numpy()
    res, i = {}, 0
    for cid in cids:
        n = len(chars[cid]["eval_strings"])
        res[cid] = [float(v) for v in p[i:i + n]]
        i += n
    return res


def _last_logits(model, x: torch.Tensor) -> torch.Tensor:
    return model(input_ids=x, attention_mask=(x != D.PAD).long(), logits_to_keep=1).logits[:, -1].float()


@torch.no_grad()
def generate_reports(model, cids, device, n_samples: int, temperature: float, seed: int,
                     batch: int = 4096) -> dict[int, tuple[list[int], list[list[int]]]]:
    """Per character: (greedy report tokens, [sampled report tokens])."""
    g = torch.Generator(device=device).manual_seed(seed)
    allowed = torch.full((model.config.vocab_size,), float("-inf"), device=device)
    allowed[[D.TOK[t] for t in D.REPORT_TOKENS] + [D.EOS]] = 0.0
    per = 1 + n_samples
    owners = np.repeat(np.asarray(cids, dtype=np.int64), per)
    greedy_row = np.tile(np.arange(per) == 0, len(cids))
    seqs = []
    for i in range(0, len(owners), batch):
        x = torch.from_numpy(D.report_prompts(owners[i:i + batch])).to(device)
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


def score_character(rule: str, strings, p_accept, reports) -> dict:
    """Same keys and values as regexintro.metrics.character_metrics, computed in one pass."""
    tgt = metrics.matches(rule, strings)
    p = np.asarray(p_accept, dtype=float)
    beh = p > 0.5
    err = beh != tgt
    ok = [r for r in reports if r is not None]
    faith, bal, tagree, exact, racc, errf = [], [], [], [], [], []
    for r in ok:
        mv = metrics.matches(r, strings)
        hit = mv == beh
        faith.append(hit.mean())
        bal.append(np.nanmean([hit[beh].mean() if beh.any() else math.nan,
                               hit[~beh].mean() if (~beh).any() else math.nan]))
        same_tgt = mv == tgt
        tagree.append(same_tgt.mean())
        exact.append(float(same_tgt.all() and grammars.equivalent(r, rule)))
        racc.append(mv.mean())
        if err.any():
            errf.append(hit[err].mean())
    mean = lambda xs: float(np.mean(xs)) if xs else math.nan  # noqa: E731
    return {
        "decision_acc": float(np.mean(beh == tgt)),
        "decision_p_target": float(np.mean(np.where(tgt, p, 1 - p))),
        "model_accept_rate": float(beh.mean()),
        "n_reports": len(reports),
        "parse_rate": len(ok) / len(reports) if reports else math.nan,
        "faithfulness": mean(faith),
        "faithfulness_balanced": mean(bal),
        "report_target_agreement": mean(tagree),
        "exact_match": mean(exact),
        "report_accept_rate": mean(racc),
        "n_error_pairs": int(err.sum()),
        "error_faithfulness": mean(errf) if err.any() else math.nan,
    }


def evaluate(model, spec, groups, device, n_samples: int = 8, temperature: float = 1.0,
             seed: int = 0, n_peers: int = 3) -> tuple[list[dict], list[dict]]:
    """Returns (per-character rows, per (group, family) summary rows)."""
    model.eval()
    chars = spec["characters"]
    cids = D.ids_of(spec, groups)
    p_accept = decision_p_accept(model, spec, cids, device)
    reps = generate_reports(model, cids, device, n_samples, temperature, seed)
    model.train()

    per, ok_of, beh_of = [], {}, {}
    for cid in cids:
        c = chars[cid]
        greedy_toks, sample_toks = reps[cid]
        parsed = [D.parse_tokens(t) for t in sample_toks]
        m = {"id": cid, "group": c["group"], "level": c["level"], "rule": c["rule"]}
        m.update(score_character(c["rule"], c["eval_strings"], p_accept[cid], parsed))
        g = D.parse_tokens(greedy_toks)
        beh = np.asarray(p_accept[cid]) > 0.5
        m["greedy_report"] = D.decode_rule(greedy_toks)
        m["greedy_faithfulness"] = metrics.agreement(g, c["eval_strings"], beh) if g is not None else math.nan
        m["greedy_exact"] = float(grammars.equivalent(g, c["rule"])) if g is not None else math.nan
        per.append(m)
        ok_of[cid] = [r for r in parsed if r is not None]
        beh_of[cid] = beh

    rng = np.random.default_rng(seed)
    by_gl = defaultdict(list)
    for m in per:
        by_gl[(m["group"], m["level"])].append(m)
    for ms in by_gl.values():
        for m in ms:
            c = chars[m["id"]]
            peers = [d["id"] for d in ms if d["id"] != c["id"]
                     and chars[d["id"]]["signature"] != c["signature"]]
            if len(peers) > n_peers:
                peers = list(rng.choice(peers, n_peers, replace=False))
            m["shuffled_faithfulness"] = _nanmean(
                [metrics.mean_over_reports(ok_of[c["id"]], chars[j]["eval_strings"], beh_of[j]) for j in peers])

    summ = []
    num = [k for k in per[0] if k not in ("id", "group", "level", "rule", "greedy_report")] if per else []
    for (grp, lv), ms in sorted(by_gl.items()):
        row = {"group": grp, "level": lv, "n": len(ms)}
        for key in num:
            row[key] = _nanmean([m[key] for m in ms])
        summ.append(row)
    return per, summ
