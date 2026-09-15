"""Combine a fresh posterior audit with resolved posterior stability."""
from __future__ import annotations


def stable_audit_acceptance(audit: dict, joint_streak: int, required: int = 2) -> bool:
    """Accept only after consecutive resolved joint-stability checks and an audit.

    The streak must come from numerically resolved comparisons of successive
    full GP posteriors. It is not the local prediction counter. Audit scheduling
    remains independent of both streaks; this function governs acceptance only.
    """
    if joint_streak < 0 or required < 1:
        raise ValueError('A nonnegative streak and positive requirement are needed')
    return bool(audit['passed'] and joint_streak >= required)
