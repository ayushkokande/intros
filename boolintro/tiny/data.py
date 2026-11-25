"""Characters, groups, rule families, vocabulary and token sequences (from-scratch model).

There are no domains or names here: every character uses the same five binary
attributes x0..x4 and is identified by its own token <cN>.

Groups
    pre_report   pretraining: choices AND self-reports
    pre_choice   pretraining: choices only (in-distribution control)
    new          finetuning only: choices only, fresh random embeddings (main test)

Rule families (see boolintro.boolfn for the degree-1 / interaction split)
    L1..L4  as in the main experiment (literal, 2-literal AND/OR, 2-term DNF, parity of 3)
    maj     majority of three literals: long formula, but fully additive
    mux     "if a then b else c": a matters, yet has zero single-attribute weight

Report formats ("worlds" for the pretraining reports)
    formula     the full rule:                        <bos> <cN> REP -> ( x0 AND NOT x3 ) OR ... <eos>
    attribute   one attribute the character cares about, drawn in proportion to its
                degree-1 weight, with its sign:        <bos> <cN> ATT -> NOT x3 <eos>
    both        either, half the time each

Decision sequence (loss only on the answer):   <bos> <cN> DEC a0..a4 | b0..b4 -> A
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from .. import boolfn, rules
from ..data import N_OBJECTS, informative_pairs  # noqa: F401  (N_OBJECTS re-exported)

K = 5
ATTRS = [f"x{i}" for i in range(K)]
GROUPS = ("pre_report", "pre_choice", "new")
FAMILIES = ("L1", "L2", "L3", "L4", "maj", "mux")
REPORT_KINDS = {"formula": ("formula",), "attribute": ("attribute",), "both": ("formula", "attribute")}

SPECIAL = ["<pad>", "<bos>", "<eos>", "DEC", "REP", "|", "->", "A", "B", "0", "1"]
RULE_TOKENS = ATTRS + ["AND", "OR", "XOR", "NOT", "(", ")"]
BASE = SPECIAL + RULE_TOKENS + ["ATT"]
LEGACY_BASE_LEN = 22  # runs made before ATT existed: character tokens start at 22
TOK = {t: i for i, t in enumerate(BASE)}
PAD, BOS, EOS = TOK["<pad>"], TOK["<bos>"], TOK["<eos>"]
PROMPT_TOKEN = {"formula": TOK["REP"], "attribute": TOK["ATT"]}
DEC_LEN = 15  # decision prompt length, up to and including "->"
REPORT_PROMPT_LEN = 4
MAX_RULE_LEN = 32

_OFFSET = len(BASE)


def use_spec(spec: dict) -> None:
    """Character tokens start after the base vocabulary the spec was built with."""
    global _OFFSET
    _OFFSET = spec.get("n_base_tokens", LEGACY_BASE_LEN)


def char_token(cid: int) -> int:
    return _OFFSET + cid


def vocab_size(spec: dict) -> int:
    return spec.get("n_base_tokens", LEGACY_BASE_LEN) + len(spec["characters"])


def report_kinds(spec: dict) -> tuple[str, ...]:
    return REPORT_KINDS[spec.get("report_format", "formula")]


# --------------------------------------------------------------------------- rules
def _lit(i: int, rng: random.Random) -> rules.Rule:
    v = ("var", i)
    return ("not", v) if rng.random() < 0.5 else v


def random_family_rule(family: str, rng: random.Random) -> rules.Rule:
    if family in ("L1", "L2", "L3", "L4"):
        return rules.random_rule(int(family[1]), rng, K)
    if family == "maj":
        a, b, c = (_lit(i, rng) for i in rng.sample(range(K), 3))
        return ("or", ("or", ("and", a, b), ("and", a, c)), ("and", b, c))
    if family == "mux":
        s, i, j = rng.sample(range(K), 3)
        return ("or", ("and", ("var", s), _lit(i, rng)), ("and", ("not", ("var", s)), _lit(j, rng)))
    raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")


def canon(rule: rules.Rule) -> str:
    """Canonical space-separated string: commutative operands sorted, chains flattened."""
    op = rule[0]
    if op == "var":
        return ATTRS[rule[1]]
    if op == "const":
        return "TRUE" if rule[1] else "FALSE"
    if op == "not":
        inner = canon(rule[1])
        return f"NOT {inner}" if rule[1][0] in ("var", "not") else f"NOT ( {inner} )"
    parts = []
    for child in rules._flatten(rule, op):
        s = canon(child)
        parts.append(f"( {s} )" if child[0] in ("and", "or", "xor") else s)
    return f" {op.upper()} ".join(sorted(parts))


def rule_tokens(rule_str: str) -> list[int]:
    return [TOK[t] for t in rule_str.split()]


def decode_rule(ids) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(BASE[i] if i < len(BASE) else "?")
    return " ".join(out)


def parse_tokens(ids) -> rules.Rule | None:
    text = decode_rule(ids)
    if not text:
        return None
    try:
        return rules.parse(text, ATTRS)
    except (rules.ParseError, RecursionError):
        return None


def attribute_statements(rule: rules.Rule) -> tuple[list[list[int]], np.ndarray]:
    """Possible attribute statements (token lists ending in <eos>) and their probabilities,
    proportional to |degree-1 weight|.  Empty when the rule has no degree-1 weight (parity)."""
    v = boolfn.valence(rule, K)
    idx = [i for i in range(K) if abs(v[i]) > 1e-9]
    stmts = [([TOK["NOT"]] if v[i] < 0 else []) + [TOK[ATTRS[i]], EOS] for i in idx]
    w = np.abs(v[idx])
    return stmts, (w / w.sum() if len(idx) else w)


# --------------------------------------------------------------------------- spec
def _held_out(rule_str: str, frac: float, seed: int) -> bool:
    if frac <= 0:
        return False
    h = int(hashlib.sha256(f"{seed}-holdout-{rule_str}".encode()).hexdigest()[:8], 16)
    return h / 2**32 < frac


def build_spec(n_pre_report: int, n_pre_choice: int, n_new: int, families: tuple[str, ...],
               seed: int, n_eval_pairs: int = 32, holdout_rule_frac: float = 0.0,
               report_format: str = "formula", report_families: tuple[str, ...] | None = None,
               n_with_reports: int | None = None) -> dict:
    """Characters with random rules.  Families are dealt out evenly within each group.

    With holdout_rule_frac > 0, a fixed fraction of rule strings (by hash) is reserved:
    pretraining characters never get them and new characters always do.

    With report_families, only pretraining characters of those families get reports; the
    others are relabelled pre_choice.  Characters and rules are identical to the
    unrestricted spec with the same seed, so runs can be compared person by person.

    With n_with_reports, only the first n pre_report characters (by id; families are dealt
    in rotation, so they stay balanced) keep their reports and the rest become pre_choice.
    Used for the "how many reports are needed" sweep: build every point of the sweep with
    the same n_pre_report and n_pre_choice=0, so all characters (and the new group) are
    identical across n.
    """
    if report_format not in REPORT_KINDS:
        raise ValueError(f"report_format must be one of {list(REPORT_KINDS)}")
    rng = random.Random(seed)
    chars = []
    for group, n in zip(GROUPS, (n_pre_report, n_pre_choice, n_new)):
        for j in range(n):
            fam = families[j % len(families)]
            want_held = group == "new"
            for _ in range(10000):
                rule = random_family_rule(fam, rng)
                s = canon(rule)
                if holdout_rule_frac <= 0 or _held_out(s, holdout_rule_frac, seed) == want_held:
                    break
            else:
                raise RuntimeError(f"no {fam} rule fits holdout_rule_frac={holdout_rule_frac}")
            pairs = informative_pairs(rule)
            rng.shuffle(pairs)
            n_eval = min(n_eval_pairs, len(pairs) // 4)
            chars.append({
                "id": len(chars), "group": group, "level": fam, "rule": s,
                "truth_table": rules.truth_table(rule, K),
                "eval_pairs": [list(p) for p in sorted(pairs[:n_eval])],
                "train_pairs": [list(p) for p in sorted(pairs[n_eval:])],
            })
    if n_with_reports is not None:
        if not 0 <= n_with_reports <= n_pre_report:
            raise ValueError(f"n_with_reports={n_with_reports} must be in [0, {n_pre_report}]")
        for c in chars:
            if c["group"] == "pre_report" and c["id"] >= n_with_reports:
                c["group"] = "pre_choice"
    if report_families is not None:
        unknown = set(report_families) - set(families)
        if unknown:
            raise ValueError(f"report_families {sorted(unknown)} not among families {list(families)}")
        for c in chars:
            if c["group"] == "pre_report" and c["level"] not in report_families:
                c["group"] = "pre_choice"
    spec = {"version": 2, "seed": seed, "families": list(families), "n_attributes": K,
            "holdout_rule_frac": holdout_rule_frac, "report_format": report_format,
            "report_families": list(report_families) if report_families is not None else None,
            "n_with_reports": n_with_reports,
            "n_base_tokens": len(BASE), "characters": chars}
    use_spec(spec)
    return spec


def char_rule(c: dict) -> rules.Rule:
    return rules.parse(c["rule"], ATTRS)


def save_spec(spec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    for c in spec["characters"]:
        assert rules.truth_table(char_rule(c), K) == c["truth_table"], c["id"]
    use_spec(spec)
    return spec


def ids_of(spec: dict, groups) -> list[int]:
    return [c["id"] for c in spec["characters"] if c["group"] in groups]


# --------------------------------------------------------------------------- sequences
_BITS = np.arange(K)


def _bits(objs: np.ndarray) -> np.ndarray:
    return ((objs[:, None] >> _BITS) & 1) + TOK["0"]


def decision_prompts(cids, a, b) -> np.ndarray:
    """[n, DEC_LEN] prompts ending in "->" (object a shown as A, b as B)."""
    cids, a, b = (np.asarray(x, dtype=np.int64) for x in (cids, a, b))
    n = len(cids)
    out = np.empty((n, DEC_LEN), dtype=np.int64)
    out[:, 0] = BOS
    out[:, 1] = char_token(0) + cids
    out[:, 2] = TOK["DEC"]
    out[:, 3:8] = _bits(a)
    out[:, 8] = TOK["|"]
    out[:, 9:14] = _bits(b)
    out[:, 14] = TOK["->"]
    return out


def report_prompts(cids, kind: str = "formula") -> np.ndarray:
    cids = np.asarray(cids, dtype=np.int64)
    out = np.empty((len(cids), REPORT_PROMPT_LEN), dtype=np.int64)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = BOS, char_token(0) + cids, PROMPT_TOKEN[kind], TOK["->"]
    return out


class Sampler:
    """Vectorised training batches: decisions for `choice_ids`, reports for `report_ids`."""

    def __init__(self, spec: dict, choice_ids, report_ids, seed: int, kinds=("formula",),
                 prefer: dict[int, dict] | None = None):
        """prefer: optional {character id: {sorted pair: chosen object}} replacing the rule's
        labels for those characters (e.g. another model's recorded choices)."""
        self.rng = np.random.default_rng(seed)
        chars = spec["characters"]
        self.choice_ids = np.asarray(choice_ids, dtype=np.int64)
        tabs = [np.asarray(chars[c]["train_pairs"], dtype=np.int64) for c in self.choice_ids]
        self.counts = np.array([len(t) for t in tabs])
        self.offsets = np.concatenate([[0], np.cumsum(self.counts)[:-1]]) if tabs else np.zeros(0, np.int64)
        self.pairs = np.concatenate(tabs) if tabs else np.zeros((0, 2), dtype=np.int64)
        pref = []
        for c, t in zip(self.choice_ids, tabs):
            tt = chars[c]["truth_table"]
            if prefer is not None and int(c) in prefer:
                pref.extend(prefer[int(c)][(int(a), int(b))] for a, b in t)
            else:
                pref.extend(a if (tt >> a) & 1 else b for a, b in t)
        self.pref = np.asarray(pref, dtype=np.int64)
        # report sources per kind: {kind: (ids, sampler(cid) -> token list)}
        self.kinds = []
        self.formula = {c: rule_tokens(chars[c]["rule"]) + [EOS] for c in report_ids} if "formula" in kinds else {}
        self.attrs = {}
        if "attribute" in kinds:
            for c in report_ids:
                st, p = attribute_statements(char_rule(chars[c]))
                if st:
                    self.attrs[c] = (st, p)
        for kind, src in (("formula", self.formula), ("attribute", self.attrs)):
            if src:
                self.kinds.append((kind, np.asarray(sorted(src), dtype=np.int64)))
        longest = max([len(s) for s in self.formula.values()] + [3])
        self.seq_len = max(DEC_LEN + 1, REPORT_PROMPT_LEN + longest)

    @property
    def has_reports(self) -> bool:
        return bool(self.kinds)

    def decisions(self, n: int):
        k = self.rng.integers(len(self.choice_ids), size=n)
        row = self.offsets[k] + (self.rng.random(n) * self.counts[k]).astype(np.int64)
        a, b = self.pairs[row, 0], self.pairs[row, 1]
        swap = self.rng.random(n) < 0.5
        a, b = np.where(swap, b, a), np.where(swap, a, b)
        label = np.where(self.pref[row] == a, TOK["A"], TOK["B"])
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        x[:, :DEC_LEN] = decision_prompts(self.choice_ids[k], a, b)
        x[:, DEC_LEN] = label
        y = np.full_like(x, -100)
        y[:, DEC_LEN] = label
        return x, y

    def reports(self, n: int):
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        y = np.full_like(x, -100)
        which = self.rng.integers(len(self.kinds), size=n)
        for i in range(n):
            kind, ids = self.kinds[which[i]]
            c = int(self.rng.choice(ids))
            if kind == "formula":
                s = self.formula[c]
            else:
                st, p = self.attrs[c]
                s = st[self.rng.choice(len(st), p=p)]
            x[i, :REPORT_PROMPT_LEN] = report_prompts([c], kind)[0]
            x[i, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + len(s)] = s
            y[i, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + len(s)] = s
        return x, y

    def batch(self, n: int, report_frac: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
        n_rep = int(round(n * report_frac)) if self.has_reports else 0
        parts = []
        if n - n_rep:
            parts.append(self.decisions(n - n_rep))
        if n_rep:
            parts.append(self.reports(n_rep))
        return parts


def collate(parts, device):
    """Right-pad (x, y) array pairs to one length and move them to `device`."""
    L = max(p[0].shape[1] for p in parts)
    xs, ys = [], []
    for x, y in parts:
        pad = L - x.shape[1]
        xs.append(np.pad(x, ((0, 0), (0, pad)), constant_values=PAD))
        ys.append(np.pad(y, ((0, 0), (0, pad)), constant_values=-100))
    x = torch.from_numpy(np.concatenate(xs)).to(device)
    y = torch.from_numpy(np.concatenate(ys)).to(device)
    return x, y, (x != PAD).long()
