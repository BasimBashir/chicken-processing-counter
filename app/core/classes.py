"""Shared detection class names for the 3-class chicken model (best.pt) and a
helper to flatten ObjectCounter's classwise output to per-class IN counts.

Kept dependency-free (no ultralytics/torch) so it imports cheaply anywhere.
"""

CLASSES = ["empty_shackles", "single_legged", "slaughtered_chicken"]


def classwise_to_counts(classwise_count: dict, classes=CLASSES) -> dict:
    """Map ObjectCounter `results.classwise_count` to a flat per-class count.

    `classwise_count` looks like ``{class_name: {"IN": n, "OUT": n}, ...}`` and
    only includes classes seen so far. The belt is one-way left->right, so each
    class's reported count is its ``IN`` value; unseen classes report 0.
    """
    out = {}
    for c in classes:
        entry = classwise_count.get(c) or {}
        out[c] = int(entry.get("IN", 0))
    return out
