"""Experiment specification: characters, hidden rules, train/eval pair splits, training stream."""

from __future__ import annotations

import json
import random
from itertools import combinations
from pathlib import Path

from . import rules
from .domains import DOMAIN_ORDER, DOMAINS, NAMES

K = 5  # attributes per object
N_OBJECTS = 2**K


def informative_pairs(rule: rules.Rule) -> list[tuple[int, int]]:
    """Unordered pairs (a < b) where exactly one object satisfies the rule."""
    return [(a, b) for a, b in combinations(range(N_OBJECTS), 2)
            if rules.evaluate(rule, a) != rules.evaluate(rule, b)]


def target_choice(rule: rules.Rule, a: int, b: int) -> int:
    """The object the character prefers (only defined on informative pairs)."""
    return a if rules.evaluate(rule, a) else b


def build_spec(
    n_characters: int = 100,
    levels: tuple[int, ...] = (1,),
    seed: int = 0,
    n_eval_pairs: int = 32,
) -> dict:
    """Characters with random rules.  Character i uses domain i % 10 and name i.

    With several levels, levels are dealt out domain by domain, so level
    counts are exactly balanced and every level sees every domain about
    equally often.
    """
    if n_characters > len(NAMES):
        raise ValueError(f"at most {len(NAMES)} characters")
    rng = random.Random(seed)
    chars = []
    n_dom = len(DOMAIN_ORDER)
    by_domain = sorted(range(n_characters), key=lambda i: (i % n_dom, i))
    level_of = {i: levels[pos % len(levels)] for pos, i in enumerate(by_domain)}
    for i in range(n_characters):
        d = i % n_dom
        level = level_of[i]
        dom = DOMAINS[DOMAIN_ORDER[d]]
        rule = rules.random_rule(level, rng, K)
        pairs = informative_pairs(rule)
        rng.shuffle(pairs)
        n_eval = min(n_eval_pairs, len(pairs) // 4)
        chars.append({
            "id": i,
            "name": NAMES[i],
            "domain": DOMAIN_ORDER[d],
            "question": dom["question"],
            "plural": dom["plural"],
            "attributes": list(dom["attributes"]),
            "level": level,
            "rule": rules.to_string(rule, dom["attributes"]),
            "truth_table": rules.truth_table(rule, K),
            "eval_pairs": [list(p) for p in sorted(pairs[:n_eval])],
            "train_pairs": [list(p) for p in sorted(pairs[n_eval:])],
        })
    return {"version": 1, "seed": seed, "levels": list(levels), "n_attributes": K, "characters": chars}


def char_rule(char: dict) -> rules.Rule:
    return rules.parse(char["rule"], char["attributes"])


def save_spec(spec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for c in spec["characters"]:  # sanity: stored rule string reproduces its truth table
        assert rules.truth_table(char_rule(c), spec["n_attributes"]) == c["truth_table"], c["name"]
    return spec


def training_stream(spec: dict, n_examples: int, seed: int, prefer: list[dict] | None = None):
    """Deterministic list of (char_index, shown_A, shown_B, label_letter).

    Characters are drawn uniformly; a training pair is drawn uniformly from
    the character's train split and shown in random order.

    By default the label is the character's hidden rule.  With `prefer` (one dict per
    character mapping a sorted pair (a, b) to the chosen object), labels come from those
    choices instead, e.g. another model's recorded behaviour (boolintro.crossmodel).
    The pairs and their order are the same either way; only the labels change.
    """
    rng = random.Random(seed)
    chars = spec["characters"]
    rule_cache = [char_rule(c) for c in chars]
    out = []
    for _ in range(n_examples):
        ci = rng.randrange(len(chars))
        a, b = rng.choice(chars[ci]["train_pairs"])
        if rng.random() < 0.5:
            a, b = b, a
        if prefer is None:
            chosen = target_choice(rule_cache[ci], a, b)
        else:
            chosen = prefer[ci][(min(a, b), max(a, b))]
        label = "A" if chosen == a else "B"
        out.append((ci, a, b, label))
    return out


def report_contexts(char: dict, n: int, seed: int) -> list[tuple[int, int]]:
    """Fixed, per-character example pairs shown inside self-report prompts."""
    rng = random.Random(f"{seed}-report-{char['id']}")
    return [tuple(rng.sample(range(N_OBJECTS), 2)) for _ in range(n)]


def observer_pairs(char: dict, n: int, seed: int) -> list[tuple[int, int]]:
    """Pairs from the train split shown (with the model's choices) to the observer."""
    rng = random.Random(f"{seed}-observer-{char['id']}")
    pairs = [tuple(p) for p in char["train_pairs"]]
    pairs = rng.sample(pairs, min(n, len(pairs)))
    return [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]
