"""Transparent English word-error scoring; no model or reference-driven rewriting.

This scorer identifies recognizer errors, not acoustic speech deletion. The tie
policy is fixed because equally optimal word alignments can have different
substitution/deletion/insertion counts even when total edit distance is equal.
"""

from __future__ import annotations

from collections import Counter
import re
import unicodedata
from typing import Iterable, Any


NORMALIZATION = (
    "Unicode NFKD, remove combining accents, uppercase; normalize curly apostrophes; "
    "tokens match [A-Z0-9]+(?:'[A-Z0-9]+)*. Other punctuation/hyphens separate words. "
    "No number expansion, spelling correction, or reference-driven rewriting."
)
ALIGNMENT_TIE_POLICY = "match, substitution, deletion, insertion"


def normalize_english(text: str) -> list[str]:
    if not isinstance(text, str):
        raise TypeError("ASR text must be a string")
    text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    text = "".join(c for c in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(c)).upper()
    return re.findall(r"[A-Z0-9]+(?:'[A-Z0-9]+)*", text)


def score_words(reference: str, hypothesis: str) -> dict[str, Any]:
    """Levenshtein alignment with explicit S/D/I counts and reference positions."""
    ref = normalize_english(reference)
    hyp = normalize_english(hypothesis)
    if not ref:
        raise ValueError("Reference must contain at least one normalized English word")
    rows, columns = len(ref) + 1, len(hyp) + 1
    costs = [[0] * columns for _ in range(rows)]
    operation = [[""] * columns for _ in range(rows)]
    for i in range(1, rows):
        costs[i][0], operation[i][0] = i, "D"
    for j in range(1, columns):
        costs[0][j], operation[0][j] = j, "I"
    for i in range(1, rows):
        for j in range(1, columns):
            if ref[i - 1] == hyp[j - 1]:
                candidates = [(costs[i - 1][j - 1], 0, "M")]
            else:
                candidates = [(costs[i - 1][j - 1] + 1, 1, "S")]
            candidates.extend(((costs[i - 1][j] + 1, 2, "D"),
                               (costs[i][j - 1] + 1, 3, "I")))
            costs[i][j], _, operation[i][j] = min(candidates)
    alignment = []
    i, j = len(ref), len(hyp)
    while i or j:
        kind = operation[i][j]
        ri = i - 1 if kind in {"M", "S", "D"} else None
        hi = j - 1 if kind in {"M", "S", "I"} else None
        alignment.append({"op": kind, "reference_index": ri, "hypothesis_index": hi,
                          "reference_word": ref[ri] if ri is not None else None,
                          "hypothesis_word": hyp[hi] if hi is not None else None})
        if kind in {"M", "S", "D"}:
            i -= 1
        if kind in {"M", "S", "I"}:
            j -= 1
    alignment.reverse()
    counts = Counter(item["op"] for item in alignment)
    n = len(ref)
    errors = counts["S"] + counts["D"] + counts["I"]
    return {"reference_normalized": " ".join(ref), "hypothesis_normalized": " ".join(hyp),
            "N": n, "S": counts["S"], "D": counts["D"], "I": counts["I"],
            "correct": counts["M"], "errors": errors, "wer": errors / n,
            "deletion_rate": counts["D"] / n, "alignment": alignment}


def corpus_score(scores: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Micro-average editing counts, never mean per-utterance WER."""
    scores = list(scores)
    total = {name: sum(s[name] for s in scores) for name in ("N", "S", "D", "I", "correct")}
    total["errors"] = total["S"] + total["D"] + total["I"]
    total["utterances"] = len(scores)
    total["wer"] = total["errors"] / total["N"] if total["N"] else None
    total["deletion_rate"] = total["D"] / total["N"] if total["N"] else None
    return total

