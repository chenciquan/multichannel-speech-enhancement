"""Greedy CTC decoding. This is an ASR proxy, not a speech-deletion detector."""
from itertools import groupby
from numbers import Integral


def greedy_ctc_tokens(indices, labels, blank=0):
    if not labels or not isinstance(blank, Integral) or not 0 <= blank < len(labels):
        raise ValueError('Invalid CTC vocabulary or blank')
    values = list(indices)
    if any(not isinstance(i, Integral) or not 0 <= i < len(labels) for i in values):
        raise ValueError('Invalid CTC token index')
    # A blank between repeats preserves a repeated letter: A,blank,A -> AA.
    return ''.join(labels[i] for i, _ in groupby(values) if i != blank).replace('|', ' ').strip()
