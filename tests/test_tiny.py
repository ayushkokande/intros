import json
import random

import numpy as np
import pytest
import torch

from boolintro import boolfn, rules
from boolintro.tiny import data as D

pytest.importorskip("transformers")


def test_canon_roundtrip_all_families():
    rng = random.Random(0)
    for fam in D.FAMILIES:
        for _ in range(200):
            r = D.random_family_rule(fam, rng)
            s = D.canon(r)
            parsed = D.parse_tokens(D.rule_tokens(s) + [D.EOS])
            assert parsed is not None and rules.equivalent(parsed, r, D.K), s
            assert len(D.rule_tokens(s)) + 1 <= D.MAX_RULE_LEN


def test_canon_is_order_free():
    a = ("and", ("var", 3), ("not", ("var", 0)))
    b = ("and", ("not", ("var", 0)), ("var", 3))
    assert D.canon(a) == D.canon(b)


def test_fourier_families():
    V = lambda i: ("var", i)  # noqa: E731
    lit = V(0)
    assert np.allclose(boolfn.degree_mass(lit), [1, 0, 0, 0, 0])
    par = ("xor", ("xor", V(0), V(1)), V(2))
    assert np.allclose(boolfn.degree_mass(par), [0, 0, 1, 0, 0])
    mux = ("or", ("and", V(0), V(1)), ("and", ("not", V(0)), V(2)))
    v = boolfn.valence(mux)
    assert abs(v[0]) < 1e-12 and v[1] > 0 and v[2] > 0  # selector has no degree-1 weight
    assert boolfn.influence(mux)[0] > 0                 # ... but does matter
    # valence equals P(true | x_i) - P(true | not x_i)
    r = ("and", V(1), ("not", V(3)))
    t = np.array([rules.evaluate(r, o) for o in range(32)])
    b3 = (np.arange(32) >> 3) & 1
    assert np.isclose(boolfn.valence(r)[3], t[b3 == 1].mean() - t[b3 == 0].mean())
    assert np.isclose(sum(c * c for c in boolfn.fourier(mux).values()), 1.0)


def test_report_stats_ties_and_signs():
    V = lambda i: ("var", i)  # noqa: E731
    target = ("and", V(0), V(1))
    long_and = ("and", ("and", ("and", V(0), V(1)), V(2)), V(3))
    pairs = [(a, b) for a in range(32) for b in range(a + 1, 32)
             if rules.evaluate(target, a) != rules.evaluate(target, b)]
    choices = [a if rules.evaluate(target, a) else b for a, b in pairs]
    st = boolfn.report_stats(long_and, target, pairs, choices)
    assert st["sign_ok"] == 1.0 and st["irr_mentioned"] == pytest.approx(2 / 3)
    assert st["decisive_acc"] == 1.0 and st["tie"] > 0.5


def test_spec_groups_families_and_holdout():
    spec = D.build_spec(60, 12, 24, ("L1", "L3", "mux"), seed=3, holdout_rule_frac=0.3)
    groups = [c["group"] for c in spec["characters"]]
    assert groups.count("pre_report") == 60 and groups.count("new") == 24
    for g in D.GROUPS:
        fams = [c["level"] for c in spec["characters"] if c["group"] == g]
        assert max(fams.count(f) for f in set(fams)) - min(fams.count(f) for f in set(fams)) <= 1
    pre = {c["rule"] for c in spec["characters"] if c["group"] != "new"}
    new = {c["rule"] for c in spec["characters"] if c["group"] == "new"}
    assert not pre & new
    for c in spec["characters"]:
        assert not set(map(tuple, c["eval_pairs"])) & set(map(tuple, c["train_pairs"]))


def test_legacy_spec_offsets():
    spec = D.build_spec(2, 1, 1, ("L1",), seed=0)
    assert D.char_token(0) == len(D.BASE)
    legacy = dict(spec)
    legacy.pop("n_base_tokens")
    D.use_spec(legacy)
    assert D.char_token(0) == D.LEGACY_BASE_LEN and D.report_kinds(legacy) == ("formula",)
    D.use_spec(spec)


def test_sampler_labels_follow_rule():
    spec = D.build_spec(6, 2, 2, ("L1", "L2"), seed=1)
    s = D.Sampler(spec, D.ids_of(spec, ("pre_report", "pre_choice")), D.ids_of(spec, ("pre_report",)), 0)
    x, y = s.decisions(500)
    chars = spec["characters"]
    for row in x:
        cid = row[1] - D.char_token(0)
        bits = row[3:8] - D.TOK["0"], row[9:14] - D.TOK["0"]
        a, b = (int(sum(int(v) << i for i, v in enumerate(bb))) for bb in bits)
        tt = chars[cid]["truth_table"]
        assert (tt >> a) & 1 != (tt >> b) & 1
        assert row[D.DEC_LEN] == (D.TOK["A"] if (tt >> a) & 1 else D.TOK["B"])
        assert chars[cid]["group"] != "new"
    xr, yr = s.reports(50)
    for row, lab in zip(xr, yr):
        cid = row[1] - D.char_token(0)
        assert chars[cid]["group"] == "pre_report" and row[2] == D.TOK["REP"]
        toks = [int(t) for t in lab if t != -100]
        assert D.decode_rule(toks) == chars[cid]["rule"]


def test_attribute_statements_follow_weights():
    spec = D.build_spec(30, 0, 0, ("L2", "L4", "mux"), seed=2, report_format="attribute")
    s = D.Sampler(spec, D.ids_of(spec, ("pre_report",)), D.ids_of(spec, ("pre_report",)), 0,
                  kinds=D.report_kinds(spec))
    chars = spec["characters"]
    assert not any(chars[c]["level"] == "L4" for c in s.attrs)  # parity: nothing to say
    xr, yr = s.reports(400)
    for row, lab in zip(xr, yr):
        c = chars[row[1] - D.char_token(0)]
        assert row[2] == D.TOK["ATT"]
        lit = D.parse_tokens([int(t) for t in lab if t != -100])
        i = next(iter(rules.variables(lit)))
        v = boolfn.valence(D.char_rule(c))
        assert abs(v[i]) > 0 and (lit[0] == "not") == (v[i] < 0)
        if c["level"] == "mux":  # the selector is never named
            sel = next(j for j in range(5) if boolfn.influence(D.char_rule(c))[j] > 0 and abs(v[j]) < 1e-9)
            assert i != sel


@pytest.mark.parametrize("fmt", ["formula", "both"])
def test_tiny_end_to_end(tmp_path, fmt):
    from boolintro.tiny import model as M
    from boolintro.tiny import probe, train

    run = tmp_path / "run"
    common = ["--run-dir", str(run), "--device", "cpu", "--batch-size", "32", "--steps", "20",
              "--eval-every", "10", "--n-samples", "2"]
    train.main(["pretrain", *common, "--n-pre-report", "12", "--n-pre-choice", "6", "--n-new", "6",
                "--families", "L1,L3,mux", "--report-format", fmt,
                "--n-layer", "1", "--n-embd", "32", "--n-head", "2"])
    before = M.load(run / "pretrain" / "model", "cpu").transformer.wte.weight.detach().clone()
    train.main(["finetune", *common])
    after = M.load(run / "ft_embed_s0" / "model", "cpu")
    w = after.transformer.wte.weight.detach()
    spec = D.load_spec(run / "spec.json")
    new_rows = [D.char_token(c) for c in D.ids_of(spec, ("new",))]
    other = torch.ones(len(w), dtype=torch.bool)
    other[new_rows] = False
    assert torch.equal(w[other], before[other])  # embed mode touches only the new rows
    assert not torch.equal(w[new_rows], before[new_rows])
    assert torch.equal(after.lm_head.weight, M.load(run / "pretrain" / "model", "cpu").lm_head.weight)
    rows = [json.loads(line) for line in (run / "ft_embed_s0" / "eval.jsonl").read_text().splitlines()]
    assert {0, 10, 20} <= {r["step"] for r in rows}
    for k in D.report_kinds(spec):
        assert f"{k}_faith" in rows[0] and f"{k}_val_corr_avg" in rows[0]
    probe.main(["--run-dir", str(run), "--device", "cpu"])
    import pandas as pd
    df = pd.read_csv(run / "probe" / "probe.csv")
    assert {"pre_report", "pre_choice", "new", "new_untrained"} <= set(df.group)
    assert {"deg1", "deg2", "deg3", "truth"} <= set(df.target)


def test_report_families_same_people_fewer_reports():
    full = D.build_spec(60, 12, 24, ("L1", "L2", "L3", "maj"), seed=5)
    part = D.build_spec(60, 12, 24, ("L1", "L2", "L3", "maj"), seed=5, report_families=("L1", "L2"))
    for a, b in zip(full["characters"], part["characters"]):
        assert (a["rule"], a["eval_pairs"], a["level"]) == (b["rule"], b["eval_pairs"], b["level"])
        if a["group"] == "pre_report":
            assert b["group"] == ("pre_report" if a["level"] in ("L1", "L2") else "pre_choice")
        else:
            assert b["group"] == a["group"]
    s = D.Sampler(part, D.ids_of(part, ("pre_report", "pre_choice")), D.ids_of(part, ("pre_report",)), 0)
    xr, _ = s.reports(200)
    assert {part["characters"][r[1] - D.char_token(0)]["level"] for r in xr} <= {"L1", "L2"}
    with pytest.raises(ValueError):
        D.build_spec(6, 0, 0, ("L1",), seed=0, report_families=("L3",))


def test_n_with_reports_same_people():
    specs = {n: D.build_spec(60, 0, 12, ("L1", "L2", "L3"), seed=4, n_with_reports=n) for n in (0, 6, 60)}
    base = specs[60]["characters"]
    for n, sp in specs.items():
        for a, b in zip(base, sp["characters"]):
            assert (a["rule"], a["eval_pairs"], a["train_pairs"]) == (b["rule"], b["eval_pairs"], b["train_pairs"])
        rep = [c for c in sp["characters"] if c["group"] == "pre_report"]
        assert len(rep) == n
        if n == 6:
            assert sorted(c["level"] for c in rep) == ["L1", "L1", "L2", "L2", "L3", "L3"]
        assert sum(c["group"] == "new" for c in sp["characters"]) == 12


def test_sweep_end_to_end(tmp_path):
    from boolintro.tiny import sweep, train
    for n in (0, 6):
        run = tmp_path / f"sweep_n{n}_s0"
        common = ["--run-dir", str(run), "--device", "cpu", "--batch-size", "32", "--steps", "10",
                  "--eval-every", "10", "--n-samples", "2"]
        train.main(["pretrain", *common, "--n-pre-report", "12", "--n-pre-choice", "6", "--n-new", "6",
                    "--families", "L1,L3", "--n-with-reports", str(n),
                    "--n-layer", "1", "--n-embd", "32", "--n-head", "2"])
        spec = json.loads((run / "spec.json").read_text())
        assert sum(c["group"] == "pre_report" for c in spec["characters"]) == n
        assert sum(c["group"] == "pre_choice" for c in spec["characters"]) == 18 - n
        train.main(["finetune", *common])
    sweep.main(["--runs", *map(str, tmp_path.glob("sweep_n*")), "--out", str(tmp_path / "out"),
                "--families", "L1,L3"])
    import pandas as pd
    df = pd.read_csv(tmp_path / "out" / "sweep.csv")
    assert set(df.N) == {0, 6} and df.faith_eff.notna().all()
    assert (tmp_path / "out" / "sweep.png").exists()
