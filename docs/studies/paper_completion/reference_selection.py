"""Retrospective checkpoint selection from saved reference-KL measurements.

This display convention does not alter acquisition or claim online convergence.
"""
from __future__ import annotations

import math

KL_LIMIT = 0.01
REQUIRED_CHECKPOINTS = 2


def symmetric_reference_kl(row: dict) -> float:
    """Average the symmetric KL estimates from the independent designs."""
    if 'symmetric_kl' in row:
        return float(row['symmetric_kl'])
    metrics = row['metrics']
    return sum((m['kl_direct_to_gp'] + m['kl_gp_to_direct']) / 2
               for m in metrics) / len(metrics)


def select_checkpoint(rows: list[dict], limit: float = KL_LIMIT,
                      required: int = REQUIRED_CHECKPOINTS) -> dict | None:
    """Return the first end of a consecutive below-threshold sequence.

Unresolved numerical estimates break the sequence, as do missing checkpoints
when the caller records them with ``numerical_pass=False``. No posterior
stability, marginal-accuracy, prediction-counter or audit criterion is used.
"""
    if not math.isfinite(limit) or limit <= 0 or required < 1:
        raise ValueError('Positive finite threshold and streak required')
    ordered = sorted(rows, key=lambda row: row['points'])
    if len({r['points'] for r in ordered}) != len(ordered):
        raise ValueError('Duplicate checkpoint counts')
    streak = 0
    for row in ordered:
        kl = symmetric_reference_kl(row)
        usable = row.get('numerical_pass', True)
        streak = streak + 1 if usable and math.isfinite(kl) and 0 <= kl < limit else 0
        if streak >= required:
            return row
    return None
