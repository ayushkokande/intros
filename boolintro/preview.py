"""Build a spec and print example prompts and rules (no GPU needed).

    python -m boolintro.preview --levels 1,2,3,4 --n 8
"""

from __future__ import annotations

import argparse
from collections import Counter

from . import data, prompts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="1,2,3,4")
    p.add_argument("--n-characters", type=int, default=100)
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--n", type=int, default=8, help="characters to print")
    a = p.parse_args(argv)
    spec = data.build_spec(a.n_characters, tuple(int(x) for x in a.levels.split(",")), a.seed)
    chars = spec["characters"]
    print("levels:", dict(Counter(c["level"] for c in chars)))
    print("domains:", dict(Counter(c["domain"] for c in chars)))
    for c in chars[: a.n]:
        print(f"  L{c['level']} {c['name']:<18} {c['domain']:<10} {c['rule']:<60} "
              f"train={len(c['train_pairs'])} eval={len(c['eval_pairs'])}")
    c = chars[0]
    a_, b_ = c["eval_pairs"][0]
    print("\n--- decision prompt ---\n" + prompts.SYSTEM + "\n\n" + prompts.decision_user(c, a_, b_))
    print("\n--- self-report prompt ---\n" + prompts.report_user(c, *data.report_contexts(c, 1, a.seed)[0]))
    ex = [(x, y, "A") for x, y in data.observer_pairs(c, 2, a.seed)]
    print("\n--- observer prompt (2 examples) ---\n" + prompts.observer_user(c, ex))


if __name__ == "__main__":
    main()
