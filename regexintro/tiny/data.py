"""Characters, groups, vocabulary and token sequences for the from-scratch regex model.

Port of boolintro.tiny.data.  Every character is identified by its own token <cN> and has a
hidden regex drawn from regexintro.grammars (levels L1..L4, random letter permutation).

Groups (as in boolintro.tiny)
    pre_report   pretraining: decisions AND self-reports (the regex)
    pre_choice   pretraining: decisions only (in-distribution control)
    new          finetuning only: decisions only, fresh random embeddings (main test)

Vocabulary
    specials     <pad> <bos> <eos> DEC REP -> Y N
    letters      a b c         shared by decision strings and reports
    regex        . | ( ) [ ] * +    (enough to write every template; no ? or {m,n})
    characters   <c0> <c1> ...      after the base vocabulary

Sequences (loss only on the answer / the report)
    decision     <bos> <cN> DEC s1 .. sL -> Y|N           (L = 1..8, one letter per token)
    report       <bos> <cN> REP -> r1 .. rk <eos>          (the canonical regex, one symbol per token)

One framing only: unlike the pretrained LLM there is no prior Yes-bias to cancel, and the
training labels are 50/50, so P(accept) = P(Y) / (P(Y) + P(N)) at the "->" position.

Canonical regex: the template instantiated with the character's letters, with the letters
inside each [...] class sorted.  Two characters get the same canonical string exactly when
they have the same language (tested), the counterpart of boolintro.tiny.data.canon.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import warnings
from itertools import permutations
from pathlib import Path

import numpy as np

from boolintro.tiny.data import collate  # noqa: F401  (generic right-pad + move to device)

from .. import data as rdata
from .. import grammars

GROUPS = ("pre_report", "pre_choice", "new")
FAMILIES = ("L1", "L2", "L3", "L4")

SPECIAL = ["<pad>", "<bos>", "<eos>", "DEC", "REP", "->", "Y", "N"]
LETTERS = list(grammars.ALPHABET)
REGEX_SYMBOLS = [".", "|", "(", ")", "[", "]", "*", "+"]
REPORT_TOKENS = LETTERS + REGEX_SYMBOLS
BASE = SPECIAL + REPORT_TOKENS
TOK = {t: i for i, t in enumerate(BASE)}
PAD, BOS, EOS = TOK["<pad>"], TOK["<bos>"], TOK["<eos>"]
ARROW, YES, NO = TOK["->"], TOK["Y"], TOK["N"]
MAX_LEN = grammars.MAX_LEN
DEC_PREFIX = 3                     # <bos> <cN> DEC
DEC_PROMPT_MAX = DEC_PREFIX + MAX_LEN + 1   # ... s1..s8 ->
REPORT_PROMPT_LEN = 4
MAX_RULE_LEN = 40                  # longest canonical regex is 32 symbols, plus <eos>

# boolintro.tiny.data.collate and boolintro.tiny.model.make_model assume these ids.
assert (PAD, BOS, EOS) == (0, 1, 2)

_OFFSET = len(BASE)


def use_spec(spec: dict) -> None:
    global _OFFSET
    _OFFSET = spec["n_base_tokens"]


def char_token(cid) -> int:
    return _OFFSET + cid


def vocab_size(spec: dict) -> int:
    return spec["n_base_tokens"] + len(spec["characters"])


def family_level(fam: str) -> int:
    return int(fam[1:])


# --------------------------------------------------------------------------- regexes
_CLASS = re.compile(r"\[([a-z]+)\]")


def canon(rx: str) -> str:
    """Canonical surface form: letters inside each [...] class sorted."""
    return _CLASS.sub(lambda m: "[" + "".join(sorted(m.group(1))) + "]", rx)


def rule_tokens(rx: str) -> list[int]:
    return [TOK[ch] for ch in rx]


def decode_rule(ids) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        out.append(BASE[i] if i < len(BASE) and BASE[i] in REPORT_TOKENS else "?")
    return "".join(out)


def parse_tokens(ids) -> str | None:
    """Generated report tokens -> a regex string, or None if it is not a valid regex."""
    text = decode_rule(ids)
    with warnings.catch_warnings():  # junk like "[[a]" triggers re's "possible nested set"
        warnings.simplefilter("ignore", FutureWarning)
        try:
            return grammars.parse(text)
        except grammars.ParseError:
            return None


def identify(level: int, rx: str) -> tuple[int, str]:
    """(template index within the level, the letter X) of a grammar drawn from that level."""
    lang = grammars.language(rx)
    for t, (template, _) in enumerate(grammars.TEMPLATES[level]):
        for letters in permutations(grammars.ALPHABET):
            if np.array_equal(grammars.language(grammars.instantiate(template, "".join(letters))), lang):
                return t, letters[0]
    raise ValueError(f"{rx!r} matches no level-{level} template")


# --------------------------------------------------------------------------- spec
def _held_out(rule: str, frac: float, seed: int) -> bool:
    if frac <= 0:
        return False
    h = int(hashlib.sha256(f"{seed}-holdout-{rule}".encode()).hexdigest()[:8], 16)
    return h / 2**32 < frac


def build_spec(n_pre_report: int, n_pre_choice: int, n_new: int, families: tuple[str, ...],
               seed: int, n_eval: int = 64, holdout_rule_frac: float = 0.0,
               report_families: tuple[str, ...] | None = None,
               n_with_reports: int | None = None) -> dict:
    """Characters with random hidden regexes; families dealt out evenly within each group.

    holdout_rule_frac, report_families and n_with_reports behave as in boolintro.tiny:
    reserve a fraction of languages (by canonical regex) for the new characters; give reports
    only to some families; or keep reports for only the first n pre_report characters (the
    "how many reports" sweep: same characters for every n).
    """
    unknown = set(families) - set(FAMILIES)
    if unknown:
        raise ValueError(f"unknown families {sorted(unknown)}; choose from {FAMILIES}")
    rng = random.Random(seed)
    chars = []
    for group, n in zip(GROUPS, (n_pre_report, n_pre_choice, n_new)):
        for j in range(n):
            fam = families[j % len(families)]
            level = family_level(fam)
            want_held = group == "new"
            for _ in range(10000):
                rx, desc = grammars.random_grammar(level, rng)
                rule = canon(rx)
                if holdout_rule_frac <= 0 or _held_out(rule, holdout_rule_frac, seed) == want_held:
                    break
            else:
                raise RuntimeError(f"no {fam} grammar fits holdout_rule_frac={holdout_rule_frac}")
            t, x = identify(level, rule)
            chars.append({
                "id": len(chars), "group": group, "level": fam, "rule": rule, "description": desc,
                "template": t, "template_id": (level - 1) * 4 + t, "x_letter": x,
                "signature": grammars.signature(rule),
                "eval_strings": rdata._sample_balanced(rule, n_eval, rng),
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
    spec = {"version": 1, "task": "regex-tiny", "seed": seed, "families": list(families),
            "alphabet": grammars.ALPHABET, "max_len": grammars.MAX_LEN, "n_eval": n_eval,
            "holdout_rule_frac": holdout_rule_frac, "report_format": "formula",
            "report_families": list(report_families) if report_families is not None else None,
            "n_with_reports": n_with_reports, "n_base_tokens": len(BASE), "characters": chars}
    use_spec(spec)
    return spec


def save_spec(spec: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec, indent=1))


def load_spec(path: Path) -> dict:
    spec = json.loads(Path(path).read_text())
    if (spec.get("alphabet"), spec.get("max_len")) != (grammars.ALPHABET, grammars.MAX_LEN):
        raise SystemExit(f"{path} was built for alphabet={spec.get('alphabet')} max_len={spec.get('max_len')}")
    for c in spec["characters"]:
        assert grammars.signature(c["rule"]) == c["signature"], c["id"]
    use_spec(spec)
    return spec


def ids_of(spec: dict, groups) -> list[int]:
    return [c["id"] for c in spec["characters"] if c["group"] in groups]


# --------------------------------------------------------------------------- sequences
_UNIV = grammars.universe()
_UINDEX = {s: i for i, s in enumerate(_UNIV)}
U_LEN = np.array([len(s) for s in _UNIV], dtype=np.int64)
U_TOK = np.full((len(_UNIV), MAX_LEN), PAD, dtype=np.int64)
for _i, _s in enumerate(_UNIV):
    U_TOK[_i, :len(_s)] = [TOK[ch] for ch in _s]


def universe_index(strings) -> np.ndarray:
    return np.array([_UINDEX[s] for s in strings], dtype=np.int64)


def decision_prompts(cids, uidx) -> tuple[np.ndarray, np.ndarray]:
    """Right-padded [n, DEC_PROMPT_MAX] prompts ending in "->", and the index of "->" per row."""
    cids, uidx = np.asarray(cids, dtype=np.int64), np.asarray(uidx, dtype=np.int64)
    n = len(cids)
    x = np.full((n, DEC_PROMPT_MAX), PAD, dtype=np.int64)
    x[:, 0], x[:, 1], x[:, 2] = BOS, char_token(0) + cids, TOK["DEC"]
    x[:, DEC_PREFIX:DEC_PREFIX + MAX_LEN] = U_TOK[uidx]
    arrow = DEC_PREFIX + U_LEN[uidx]
    x[np.arange(n), arrow] = ARROW
    return x, arrow


def report_prompts(cids) -> np.ndarray:
    cids = np.asarray(cids, dtype=np.int64)
    out = np.empty((len(cids), REPORT_PROMPT_LEN), dtype=np.int64)
    out[:, 0], out[:, 1], out[:, 2], out[:, 3] = BOS, char_token(0) + cids, TOK["REP"], ARROW
    return out


class Sampler:
    """Vectorised training batches: decisions for `choice_ids`, reports for `report_ids`.

    Decisions follow regexintro.data.training_stream: character uniform, label 50/50, then a
    length-stratified string (length uniform over the lengths where that label exists in the
    character's train pool, i.e. the universe minus its eval strings).
    """

    def __init__(self, spec: dict, choice_ids, report_ids, seed: int):
        self.rng = np.random.default_rng(seed)
        chars = spec["characters"]
        self.choice_ids = np.asarray(choice_ids, dtype=np.int64)
        n = len(self.choice_ids)
        self.start = np.zeros((n, 2, MAX_LEN), dtype=np.int64)
        self.count = np.zeros((n, 2, MAX_LEN), dtype=np.int64)
        pools, off = [], 0
        for k, cid in enumerate(self.choice_ids):
            c = chars[cid]
            lang = grammars.language(c["rule"])
            free = np.ones(len(_UNIV), dtype=bool)
            free[universe_index(c["eval_strings"])] = False
            for lab in (0, 1):
                idx = np.flatnonzero(free & (lang == bool(lab)))  # universe order: by length
                cnt = np.bincount(U_LEN[idx] - 1, minlength=MAX_LEN)
                self.count[k, lab] = cnt
                self.start[k, lab] = off + np.concatenate([[0], np.cumsum(cnt)[:-1]])
                pools.append(idx)
                off += len(idx)
        self.pool = np.concatenate(pools) if pools else np.zeros(0, dtype=np.int64)
        self.report_ids = np.asarray(sorted(report_ids), dtype=np.int64)
        toks = [rule_tokens(chars[c]["rule"]) + [EOS] for c in self.report_ids]
        self.rep_len = np.array([len(t) for t in toks], dtype=np.int64)
        width = max([len(t) for t in toks] + [1])
        self.rep_mat = np.full((len(toks), width), PAD, dtype=np.int64)
        for i, t in enumerate(toks):
            self.rep_mat[i, :len(t)] = t
        self.seq_len = max(DEC_PROMPT_MAX + 1, REPORT_PROMPT_LEN + width)

    @property
    def has_reports(self) -> bool:
        return len(self.report_ids) > 0

    def decisions(self, n: int):
        rng = self.rng
        k = rng.integers(len(self.choice_ids), size=n)
        lab = (rng.random(n) < 0.5).astype(np.int64)
        cnt = self.count[k, lab]                                   # [n, MAX_LEN]
        length = np.argmax(rng.random((n, MAX_LEN)) * (cnt > 0), 1)
        rows = np.arange(n)
        pick = self.start[k, lab, length] + (rng.random(n) * cnt[rows, length]).astype(np.int64)
        prompt, arrow = decision_prompts(self.choice_ids[k], self.pool[pick])
        ans = np.where(lab == 1, YES, NO)
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        x[:, :DEC_PROMPT_MAX] = prompt
        x[rows, arrow + 1] = ans
        y = np.full_like(x, -100)
        y[rows, arrow + 1] = ans
        return x, y

    def reports(self, n: int):
        j = self.rng.integers(len(self.report_ids), size=n)
        x = np.full((n, self.seq_len), PAD, dtype=np.int64)
        x[:, :REPORT_PROMPT_LEN] = report_prompts(self.report_ids[j])
        w = self.rep_mat.shape[1]
        x[:, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + w] = self.rep_mat[j]
        y = np.full_like(x, -100)
        body = x[:, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + w]
        y[:, REPORT_PROMPT_LEN:REPORT_PROMPT_LEN + w] = np.where(body == PAD, -100, body)
        return x, y

    def batch(self, n: int, report_frac: float = 0.0) -> list[tuple[np.ndarray, np.ndarray]]:
        n_rep = int(round(n * report_frac)) if self.has_reports else 0
        parts = []
        if n - n_rep:
            parts.append(self.decisions(n - n_rep))
        if n_rep:
            parts.append(self.reports(n_rep))
        return parts
