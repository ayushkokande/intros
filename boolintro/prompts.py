"""Prompt templates, following Atkinson et al. (Appendix A.2), adapted to boolean attributes."""

from __future__ import annotations

from typing import Sequence

SYSTEM = "Your job is to make hypothetical decisions on behalf of different people or characters."

DECISION_HEADER = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you think '
    'Option B is better. Never respond with anything except "A" or "B":'
)

REPORT_HEADER = (
    "[INTROSPECTION TASK] Write the rule you use to make this kind of decision. Your rule is a "
    "yes/no condition on the five attributes: you prefer the option for which the condition is "
    "true. Write it as a boolean formula over the attribute names ({names}) using AND, OR, NOT, "
    "XOR and parentheses. Respond with the formula only, nothing else. (Do not report your "
    "decision itself.):"
)

OBSERVER_HEADER = (
    "[PREDICTION TASK] Below are decisions that {name} made between pairs of {plural}. "
    "{name}'s choices follow a yes/no condition on the five attributes: {name} prefers the option "
    "for which the condition is true. Write that condition as a boolean formula over the attribute "
    "names ({names}) using AND, OR, NOT, XOR and parentheses. Respond with the formula only, "
    "nothing else."
)


def describe(obj: int, attributes: Sequence[str]) -> str:
    return "\n".join(f"{a}: {'yes' if (obj >> i) & 1 else 'no'}" for i, a in enumerate(attributes))


def scenario(char: dict, a: int, b: int) -> str:
    attrs = char["attributes"]
    return (
        f"Imagine you are {char['name']}. {char['question']}\n"
        f"A:\n{describe(a, attrs)}\n\nB:\n{describe(b, attrs)}"
    )


def decision_user(char: dict, a: int, b: int) -> str:
    return f"{DECISION_HEADER}\n\n{scenario(char, a, b)}"


def report_user(char: dict, a: int, b: int) -> str:
    header = REPORT_HEADER.format(names=", ".join(char["attributes"]))
    return f"{header}\n\n{scenario(char, a, b)}"


def observer_user(char: dict, examples: Sequence[tuple[int, int, str]]) -> str:
    """examples: (object shown as A, object shown as B, chosen letter)."""
    header = OBSERVER_HEADER.format(
        name=char["name"], plural=char["plural"], names=", ".join(char["attributes"])
    )
    attrs = char["attributes"]
    blocks = [
        f"Decision {n}:\nA:\n{describe(a, attrs)}\n\nB:\n{describe(b, attrs)}\n\n{char['name']} chose: {c}"
        for n, (a, b, c) in enumerate(examples, 1)
    ]
    return header + "\n\n" + "\n\n".join(blocks) + f"\n\nWhat is {char['name']}'s rule?"


# --------------------------------------------------------------------------- conflict test
# A record of past decisions placed before the task (see boolintro/conflict.py).
# "self": framed as the model's own memory; "other": another assistant's decisions;
# "unlabeled": the same decisions as plain examples, with no one named as the chooser.
RECORD_HEADERS = {
    "self": "[MEMORY] Notes saved from your earlier sessions. In those sessions you made these "
            "decisions on behalf of {name}:",
    "other": "[NOTES] Another assistant made these decisions on behalf of {name}:",
    "unlabeled": "[EXAMPLES] Example decisions between pairs of {plural}:",
}
RECORD_CHOOSER = {"self": "You chose", "other": "It chose", "unlabeled": "Chosen"}

DETECT_QUESTIONS = {
    # First person: was this record really yours?
    "self": 'Did you actually make the decisions in these notes? Respond with "Yes" or "No" only.',
    # Third person: the same judgement without the "you" framing.
    "third": 'Are the decisions above consistent with how {name} chooses? Respond with "Yes" or '
             '"No" only.',
}
DETECT_RECORD_FRAME = {"self": "self", "third": "unlabeled_named"}


def record_block(char: dict, examples: Sequence[tuple[int, int, str]], frame: str) -> str:
    """examples: (object shown as A, object shown as B, chosen letter)."""
    attrs = char["attributes"]
    if frame == "unlabeled_named":  # third-person detection: decisions attributed to the character
        header, chooser = f"[NOTES] Decisions recorded for {char['name']}:", f"{char['name']} chose"
    else:
        header = RECORD_HEADERS[frame].format(name=char["name"], plural=char["plural"])
        chooser = RECORD_CHOOSER[frame]
    blocks = [f"Decision {n}:\nA:\n{describe(a, attrs)}\n\nB:\n{describe(b, attrs)}\n\n{chooser}: {c}"
              for n, (a, b, c) in enumerate(examples, 1)]
    return header + "\n\n" + "\n\n".join(blocks)


def with_record(record: str | None, task: str) -> str:
    return task if not record else f"{record}\n\n[END OF NOTES]\n\n{task}"


def detect_user(char: dict, examples: Sequence[tuple[int, int, str]], framing: str) -> str:
    rec = record_block(char, examples, DETECT_RECORD_FRAME[framing])
    return f"{rec}\n\n[END OF NOTES]\n\n" + DETECT_QUESTIONS[framing].format(name=char["name"])
