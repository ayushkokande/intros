import math
from types import SimpleNamespace

from boolintro import conflict, data, prompts, rules

NAMES = ["balcony", "pool", "noisy", "near_center", "pet_friendly"]


def P(s):
    return rules.parse(s, NAMES)


def test_disagreement_pairs_are_opposite_under_both_rules():
    r, alt = P("pool"), P("noisy")
    dis = conflict.disagreement_pairs(r, alt)
    # pool=1,noisy=0 (8 objects) vs pool=0,noisy=1 (8 objects)
    assert len(dis) == 64
    for a, b in dis:
        assert data.target_choice(r, a, b) != data.target_choice(alt, a, b)
    # Half of the jointly informative pairs disagree for two different literals.
    assert len(dis) / len(conflict.joint_pairs(r, alt)) == 0.5


def _char(level, rule_str, cid=0):
    return {"id": cid, "name": "Prometheus", "level": level, "rule": rule_str, "attributes": NAMES,
            "plural": "hotels", "question": "Which hotel?"}


def test_pick_conflict_rule_excludes_R_and_not_R():
    for level, rs in [(1, "NOT pool"), (2, "balcony AND NOT noisy"), (2, "pool OR pet_friendly")]:
        c = _char(level, rs)
        r = P(rs)
        for seed in range(20):
            alt, dis, frac = conflict.pick_conflict_rule(c, r, need=24, min_frac=0.4, seed=seed)
            tt, ta = rules.truth_table(r), rules.truth_table(alt)
            assert ta not in (tt, tt ^ conflict.FULL_TABLE)
            assert len(dis) >= 24 and frac >= 0.4


def test_build_items_nested_records_and_disjoint_eval():
    args = SimpleNamespace(min_eval=8, n_eval=24, min_disagree=0.4, n_reports=4)
    c = _char(2, "balcony AND NOT noisy")
    it = conflict.build_items(c, [4, 8, 16], args, seed=1)
    assert len(it["record_pairs"]) == 16
    rec = {tuple(sorted(p)) for p in it["record_pairs"]}
    assert not rec & set(it["eval_pairs"])
    assert len(it["eval_pairs"]) >= 8


def test_score_context_signs():
    r, alt = P("pool"), P("noisy")
    pairs = conflict.disagreement_pairs(r, alt)[:10]
    # Model that follows R perfectly, reports R.
    p_first = [1.0 if rules.evaluate(r, a) else 0.0 for a, _ in pairs]
    m, _ = conflict.score_context(r, alt, pairs, p_first, ["pool"] * 3, NAMES)
    assert m["behavior_shift"] == -1 and m["report_shift"] == -1 and m["faith_in_context"] == 1
    # Model that follows R', reports R'.
    p_first = [1.0 if rules.evaluate(alt, a) else 0.0 for a, _ in pairs]
    m, _ = conflict.score_context(r, alt, pairs, p_first, ["noisy", "noisy", "garbage ???"], NAMES)
    assert m["behavior_shift"] == 1 and m["report_shift"] == 1 and math.isclose(m["parse_rate"], 2 / 3)
    assert m["exact_alt"] == 1 and m["exact_R"] == 0
    # A report that ties everywhere scores 0.
    m, _ = conflict.score_context(r, alt, pairs, p_first, ["balcony AND NOT balcony"], NAMES)
    assert m["report_shift"] == 0 and m["report_tie"] == 1


def test_prompts_render():
    c = _char(1, "pool")
    ex = [(3, 5, "A"), (1, 2, "B")]
    for frame in ("self", "other", "unlabeled"):
        txt = prompts.with_record(prompts.record_block(c, ex, frame), prompts.decision_user(c, 3, 5))
        assert "Decision 2:" in txt and "[DECISION TASK]" in txt
    assert "You chose: A" in prompts.record_block(c, ex, "self")
    assert "Prometheus chose: B" in prompts.detect_user(c, ex, "third")
    assert prompts.with_record(None, "x") == "x"
