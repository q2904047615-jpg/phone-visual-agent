from __future__ import annotations


def reject_if(condition: object, error: Exception) -> None:
    """Raise a prepared contract error when an invariant is violated."""

    if condition:
        raise error
