"""Backward-compatible compute_lambda entry point.

The project historically imported ``utils.compute_lambda`` from multiple
places. The implementation file was renamed to ``compute_lambda_usage.py``,
but not all callers were updated. Re-export the function here so the rest of
the codebase continues to use a stable import path.
"""

from .compute_lambda_usage import compute_lambda

__all__ = ["compute_lambda"]
