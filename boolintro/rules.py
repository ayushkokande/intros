"""Boolean rules over binary attributes: generation, evaluation, printing, parsing.

A rule is a small AST of tuples:
    ("var", i) | ("const", bool) | ("not", x) | ("and", a, b) | ("or", a, b) | ("xor", a, b)

Objects are integers in [0, 2**k): bit i is attribute i (1 = yes).

Difficulty levels (proposal table):
    1  one literal                     pool              / NOT pool
    2  AND or OR of two literals       balcony AND NOT noisy
    3  OR of two ANDs (2-term DNF)     (balcony AND NOT noisy) OR (pool AND pet_friendly)
    4  parity of three                 balcony XOR pool XOR noisy  (one literal may be negated)

Every literal gets a random sign so that common-sense priors about attributes
("a pool is good") cannot predict the rule.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

Rule = tuple

LEVELS = (1, 2, 3, 4)


# --------------------------------------------------------------------------- eval
def evaluate(rule: Rule, obj: int) -> bool:
    op = rule[0]
    if op == "var":
        return bool((obj >> rule[1]) & 1)
    if op == "const":
        return bool(rule[1])
    if op == "not":
        return not evaluate(rule[1], obj)
    a, b = evaluate(rule[1], obj), evaluate(rule[2], obj)
    if op == "and":
        return a and b
    if op == "or":
        return a or b
    if op == "xor":
        return a != b
    raise ValueError(f"unknown op {op!r}")


def truth_table(rule: Rule, k: int = 5) -> int:
    """Bitmask over all 2**k objects: bit o is rule(o)."""
    return sum(1 << o for o in range(2**k) if evaluate(rule, o))


def equivalent(r1: Rule, r2: Rule, k: int = 5) -> bool:
    return truth_table(r1, k) == truth_table(r2, k)


def variables(rule: Rule) -> set[int]:
    if rule[0] == "var":
        return {rule[1]}
    if rule[0] == "const":
        return set()
    return set().union(*(variables(c) for c in rule[1:]))


# --------------------------------------------------------------------------- generation
def _lit(i: int, rng: random.Random) -> Rule:
    v = ("var", i)
    return ("not", v) if rng.random() < 0.5 else v


def random_rule(level: int, rng: random.Random, k: int = 5) -> Rule:
    """Sample a random rule of a difficulty level over k attributes.

    All variables in a rule are distinct, so the rule depends on each of them
    and is never constant.
    """
    if level == 1:
        (i,) = rng.sample(range(k), 1)
        return _lit(i, rng)
    if level == 2:
        i, j = rng.sample(range(k), 2)
        op = rng.choice(["and", "or"])
        return (op, _lit(i, rng), _lit(j, rng))
    if level == 3:
        i, j, m, n = rng.sample(range(k), 4)
        return ("or", ("and", _lit(i, rng), _lit(j, rng)), ("and", _lit(m, rng), _lit(n, rng)))
    if level == 4:
        i, j, m = sorted(rng.sample(range(k), 3))
        lits = [("var", i), ("var", j), ("var", m)]
        if rng.random() < 0.5:  # parity or its complement
            p = rng.randrange(3)
            lits[p] = ("not", lits[p])
        return ("xor", ("xor", lits[0], lits[1]), lits[2])
    raise ValueError(f"level must be one of {LEVELS}, got {level}")


# --------------------------------------------------------------------------- printing
_PREC = {"or": 1, "xor": 2, "and": 3}


def _flatten(rule: Rule, op: str) -> list[Rule]:
    if rule[0] == op:
        return _flatten(rule[1], op) + _flatten(rule[2], op)
    return [rule]


def to_string(rule: Rule, names: Sequence[str]) -> str:
    op = rule[0]
    if op == "var":
        return names[rule[1]]
    if op == "const":
        return "TRUE" if rule[1] else "FALSE"
    if op == "not":
        inner = to_string(rule[1], names)
        return f"NOT {inner}" if rule[1][0] in ("var", "const", "not") else f"NOT ({inner})"
    parts = []
    for child in _flatten(rule, op):
        s = to_string(child, names)
        if child[0] in _PREC:  # always bracket a different binary op, for readability
            s = f"({s})"
        parts.append(s)
    return f" {op.upper()} ".join(parts)


# --------------------------------------------------------------------------- parsing
class ParseError(ValueError):
    pass


_SYMBOLS = {
    "&&": "and", "&": "and", "∧": "and", "*": "and",
    "||": "or", "|": "or", "∨": "or", "+": "or",
    "^": "xor", "⊕": "xor", "!=": "neq",
    "!": "not", "~": "not", "¬": "not",
    "==": "eq", "=": "eq", ":": "eq",
}
_TOKEN_RE = re.compile(r"\s*(&&|\|\||==|!=|[&|^!~¬∧∨⊕*+=:()]|[A-Za-z_][A-Za-z0-9_]*|\d+)")
_TRUE_WORDS = {"yes", "true", "1"}
_FALSE_WORDS = {"no", "false", "0"}


def _normalise(text: str, names: Sequence[str]) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    text = text.replace("`", "").replace("**", "").replace('"', "").replace("'", "")
    text = re.sub(r"^\s*(rule|formula|answer)\s*:\s*", "", text, flags=re.I)
    text = text.strip().rstrip(".").strip()
    # Accept attribute names written with spaces or hyphens ("near center").
    for name in sorted(names, key=len, reverse=True):
        pattern = r"\b" + r"[\s_\-]+".join(map(re.escape, name.split("_"))) + r"\b"
        text = re.sub(pattern, name, text, flags=re.I)
    return text


def _tokenize(text: str) -> list[str]:
    pos, out = 0, []
    while pos < len(text):
        if text[pos:].strip() == "":
            break
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"bad character at {pos}: {text[pos:pos + 10]!r}")
        out.append(m.group(1))
        pos = m.end()
    return out


class _Parser:
    # Precedence: NOT > AND > XOR > OR
    def __init__(self, tokens: list[str], names: Sequence[str]):
        self.toks = tokens
        self.i = 0
        self.index = {n.lower(): j for j, n in enumerate(names)}

    def peek(self) -> str | None:
        if self.i >= len(self.toks):
            return None
        t = self.toks[self.i]
        return _SYMBOLS.get(t, t.lower())

    def take(self) -> str:
        t = self.peek()
        self.i += 1
        return t

    def parse(self) -> Rule:
        r = self.p_or()
        if self.peek() is not None:
            raise ParseError(f"trailing tokens: {self.toks[self.i:]}")
        return r

    def _binary(self, op: str, sub):
        left = sub()
        while self.peek() == op:
            self.take()
            left = (op, left, sub())
        return left

    def p_or(self):
        return self._binary("or", self.p_xor)

    def p_xor(self):
        return self._binary("xor", self.p_and)

    def p_and(self):
        return self._binary("and", self.p_not)

    def p_not(self):
        if self.peek() == "not":
            self.take()
            return ("not", self.p_not())
        return self.p_atom()

    def p_atom(self):
        t = self.take()
        if t is None:
            raise ParseError("unexpected end of formula")
        if t == "(":
            r = self.p_or()
            if self.take() != ")":
                raise ParseError("missing )")
            return self._comparison(r)
        if t in self.index:
            return self._comparison(("var", self.index[t]))
        if t in ("true", "false"):
            return ("const", t == "true")
        raise ParseError(f"unknown token {t!r}")

    def _comparison(self, r: Rule) -> Rule:
        """Postfix comparisons.

        `pool: yes`, `pool = no`, `pool is yes`, `pool != 1` compare with a constant;
        `(x) == (y)` / `(x) != (y)` between two formulas mean XNOR / XOR.
        """
        nxt = self.peek()
        if nxt in ("eq", "neq", "is"):
            self.take()
            negate = nxt == "neq"
            if self.peek() == "not":  # "is not"
                self.take()
                negate = not negate
            val = self.peek()
            if val in _TRUE_WORDS or val in _FALSE_WORDS:
                self.take()
                if (val in _TRUE_WORDS) == negate:
                    r = ("not", r)
            else:
                other = self.p_not()
                r = ("xor", r, other) if negate else ("not", ("xor", r, other))
        return r


def parse(text: str, names: Sequence[str]) -> Rule:
    """Parse a model-written formula over attribute `names`.  Raises ParseError."""
    norm = _normalise(text, names)
    if not norm:
        raise ParseError("empty")
    return _Parser(_tokenize(norm), names).parse()


_LISTING_RE = re.compile(r"^\s*[A-Za-z_][\w ]*\s*:\s*(yes|no)\s*$", re.I)


def parse_report(text: str, names: Sequence[str]) -> Rule | None:
    """Lenient parse: whole text, then the first non-empty line.  None on failure.

    The first-line fallback (for "formula\n\nexplanation") is not used when the
    text looks like an echoed option listing ("pool: yes\nnoisy: no ..."), which
    would otherwise be misread as the rule given on its first line.
    """
    candidates = [text]
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("```")]
    if len(lines) > 1 and sum(bool(_LISTING_RE.match(ln)) for ln in lines) < 2:
        candidates.append(lines[0])
    for cand in candidates:
        try:
            return parse(cand, names)
        except (ParseError, RecursionError):
            continue
    return None
