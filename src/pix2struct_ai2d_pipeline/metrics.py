"""Multiple-choice scoring for diagram question answering: accuracy with unmatched answers counted wrong, the
unmatched rate, the chance level, a per-category breakdown, and two non-neural baselines that answer without
looking at the diagram.

An answer is correct when the option its generated text matches (normalised exact match, `match_option`)
is the record's correct option; an answer that matches no option is wrong. `chance` is the mean of
1 / n_options over the scored questions — what uniform guessing would score. The **position-prior baseline**
answers every test question with the option position most often correct in the training split (clamped to
the question's option count); the **longest-option baseline** answers with the longest option text (the
first on ties). Both see the question text or the option layout only, never the diagram, so a system that
does not beat them has not shown it reads the diagram.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

METRIC_DEFINITIONS: dict[str, str] = {
    "accuracy": (
        "fraction of questions whose matched option is the correct one; unmatched answers count as wrong"
    ),
    "unmatched_rate": "fraction of questions whose generated text matches none of the options",
    "chance": "mean of 1 / n_options over the scored questions (uniform guessing)",
    "by_category": "accuracy per record category (`letter-label` / `text-option` in the sample)",
}


def mcq_metrics(
    predicted: Sequence[int | None],
    correct: Sequence[int],
    n_options: Sequence[int],
    *,
    categories: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Accuracy, unmatched rate, chance and per-category accuracy for aligned predictions."""
    n = len(predicted)
    if n == 0 or len(correct) != n or len(n_options) != n:
        raise ValueError("predicted, correct and n_options must be non-empty and aligned")
    if categories is not None and len(categories) != n:
        raise ValueError("categories must align with the predictions")
    for gold, count in zip(correct, n_options, strict=True):
        if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < count:
            raise ValueError("each correct index must be a valid zero-based option index")
    hits = [p is not None and p == g for p, g in zip(predicted, correct, strict=True)]
    groups: dict[str, list[bool]] = defaultdict(list)
    for hit, category in zip(hits, categories or ["all"] * n, strict=True):
        groups[str(category)].append(hit)
    return {
        "n": n,
        "accuracy": sum(hits) / n,
        "unmatched_rate": sum(p is None for p in predicted) / n,
        "chance": sum(1.0 / c for c in n_options) / n,
        "by_category": {
            name: {"n": len(values), "accuracy": sum(values) / len(values)}
            for name, values in sorted(groups.items())
        },
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _score(
    name: str, note: str, predicted: list[int], test: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    metrics = mcq_metrics(
        predicted,
        [int(r["answer"]) for r in test],
        [len(r["options"]) for r in test],
        categories=[str(r.get("category", "other")) for r in test],
    )
    return {**metrics, "baseline": name, "note": note}


def position_prior_baseline(
    train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Answer every test question with the option position most often correct in `train`."""
    if not train or not test:
        raise ValueError("train and test must be non-empty")
    counts = Counter(int(r["answer"]) for r in train)
    position = min(counts, key=lambda k: (-counts[k], k))
    predicted = [min(position, len(r["options"]) - 1) for r in test]
    note = f"always option {position + 1} (most frequent correct position in training)"
    return {**_score("position-prior", note, predicted, test), "position": position}


def longest_option_baseline(test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Answer every test question with its longest option text (the first on ties)."""
    if not test:
        raise ValueError("test must be non-empty")
    predicted = [max(range(len(r["options"])), key=lambda k, r=r: (len(r["options"][k]), -k)) for r in test]
    note = "the longest option text, without looking at the diagram"
    return _score("longest-option", note, predicted, test)
