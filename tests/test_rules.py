import random

import pytest

from boolintro import data, metrics, rules

NAMES = ["balcony", "pool", "noisy", "near_center", "pet_friendly"]


@pytest.mark.parametrize("level", rules.LEVELS)
def test_random_rules_roundtrip_and_nontrivial(level):
    rng = random.Random(level)
    for _ in range(200):
        r = rules.random_rule(level, rng)
        tt = rules.truth_table(r)
        assert 0 < tt < 2**32 - 1  # never constant
        s = rules.to_string(r, NAMES)
        assert rules.truth_table(rules.parse(s, NAMES)) == tt, s
        assert len(rules.variables(r)) == {1: 1, 2: 2, 3: 4, 4: 3}[level]


def test_level4_is_parity():
    rng = random.Random(0)
    for _ in range(50):
        r = rules.random_rule(4, rng)
        n_true = bin(rules.truth_table(r)).count("1")
        assert n_true == 16


@pytest.mark.parametrize("text,expected", [
    ("pool", "pool"),
    ("NOT noisy", "NOT noisy"),
    ("balcony AND NOT noisy", "balcony AND NOT noisy"),
    ("(balcony and not noisy) or (pool and pet_friendly)", "(balcony AND NOT noisy) OR (pool AND pet_friendly)"),
    ("balcony XOR pool XOR noisy", "balcony XOR pool XOR noisy"),
    ("balcony ^ pool ^ noisy", "balcony XOR pool XOR noisy"),
    ("balcony && !noisy", "balcony AND NOT noisy"),
    ("`near center AND pet-friendly`", "near_center AND pet_friendly"),
    ("Rule: pool: yes AND noisy: no.", "pool AND NOT noisy"),
    ("pool = no OR noisy == 1", "NOT pool OR noisy"),
    ("pool is not yes", "NOT pool"),
    ("a OR b AND c".replace("a", "pool").replace("b", "noisy").replace("c", "balcony"),
     "pool OR (noisy AND balcony)"),
    ("NOT (pool OR noisy)", "NOT pool AND NOT noisy"),
    ('(balcony == "yes") AND (pool == "no")', "balcony AND NOT pool"),
    ("(balcony == yes) != (pool == yes)", "balcony XOR pool"),
    ("(balcony) == (pool)", "NOT (balcony XOR pool)"),
    ("¬balcony ∧ (pool ∨ noisy)", "NOT balcony AND (pool OR noisy)"),
])
def test_parse(text, expected):
    got = rules.parse(text, NAMES)
    assert rules.equivalent(got, rules.parse(expected, NAMES))


@pytest.mark.parametrize("text", ["", "pool AND", "pool AND garage", "I prefer pools", "((pool)"])
def test_parse_failures(text):
    assert rules.parse_report(text, NAMES) is None


def test_option_listing_is_not_a_rule():
    echoed = "balcony: no\npool: no\nnoisy: no\nnear_center: no\npet_friendly: yes"
    assert rules.parse_report(echoed, NAMES) is None


def test_parse_report_first_line():
    r = rules.parse_report("pool AND noisy\n\nThis is because I like pools.", NAMES)
    assert rules.equivalent(r, rules.parse("pool AND noisy", NAMES))


def test_pair_score():
    r = rules.parse("pool", NAMES)
    a, b = 0b00010, 0b00000  # a has a pool
    assert metrics.pair_score(r, a, b, a) == 1.0
    assert metrics.pair_score(r, a, b, b) == 0.0
    assert metrics.pair_score(r, a, a | 1, a) == 0.5  # both satisfy -> tie


def test_character_metrics_perfect_and_error_pairs():
    target = rules.parse("pool AND NOT noisy", NAMES)
    pairs = data.informative_pairs(target)[:20]
    # model behaves exactly like target
    p = [1.0 if rules.evaluate(target, a) else 0.0 for a, b in pairs]
    m = metrics.character_metrics(target, pairs, p, [target, None])
    assert m["decision_acc"] == 1.0 and m["faithfulness"] == 1.0 and m["exact_match"] == 1.0
    assert m["parse_rate"] == 0.5 and m["n_error_pairs"] == 0
    # model behaves like "noisy"; report "noisy" is faithful to behaviour, not target
    beh = rules.parse("noisy", NAMES)
    pairs = [(a, b) for a, b in data.informative_pairs(target) if rules.evaluate(beh, a) != rules.evaluate(beh, b)]
    p = [1.0 if rules.evaluate(beh, a) else 0.0 for a, b in pairs]
    m = metrics.character_metrics(target, pairs, p, [beh])
    assert m["faithfulness"] == 1.0
    assert m["n_error_pairs"] > 0 and m["error_faithfulness"] == 1.0
    assert m["report_target_agreement"] < 1.0


def test_spec_balanced_and_disjoint():
    spec = data.build_spec(100, (1, 2, 3, 4), seed=1)
    from collections import Counter
    assert Counter(c["level"] for c in spec["characters"]) == {1: 25, 2: 25, 3: 25, 4: 25}
    for c in spec["characters"]:
        tr = {tuple(p) for p in c["train_pairs"]}
        ev = {tuple(p) for p in c["eval_pairs"]}
        assert not tr & ev and len(ev) >= 16
        r = data.char_rule(c)
        assert all(rules.evaluate(r, a) != rules.evaluate(r, b) for a, b in tr | ev)
    stream = data.training_stream(spec, 500, seed=3)
    assert stream == data.training_stream(spec, 500, seed=3)
    assert 0.4 < sum(lab == "A" for *_, lab in stream) / 500 < 0.6


def test_score_step_shuffled_and_observer():
    from boolintro.evaluate import score_step

    spec = data.build_spec(20, (2,), seed=4)
    chars = spec["characters"]
    # model behaves exactly per target; every report states the true rule
    p_first = [[1.0 if rules.evaluate(data.char_rule(c), a) else 0.0 for a, b in c["eval_pairs"]]
               for c in chars]
    raw = {ci: [c["rule"], "garbage"] for ci, c in enumerate(chars)}
    obs = {ci: ["TRUE"] for ci in range(len(chars))}
    per_char, _ = score_step(chars, p_first, raw, obs)
    for m in per_char:
        assert m["faithfulness"] == 1.0 and m["parse_rate"] == 0.5 and m["exact_match"] == 1.0
        assert m["shuffled_faithfulness"] < 1.0  # peers in same domain have other rules
        assert m["observer_faithfulness"] == 0.5  # constant rule ties every pair
