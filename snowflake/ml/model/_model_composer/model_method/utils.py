from __future__ import annotations

from typing import Any, Mapping, Optional


def determine_explain_case_sensitive_from_method_options(
    method_options: Mapping[str, Optional[Mapping[str, Any]]],
    target_method: str,
) -> bool:
    """Determine explain method case sensitivity from related predict methods.

    Args:
        method_options: Mapping from method name to its options. Each option may
            contain ``"case_sensitive"`` to indicate SQL identifier sensitivity.
        target_method: The target method name being resolved (e.g., an ``explain_*``
            method).

    Returns:
        True if the explain method should be treated as case sensitive; otherwise False.
    """
    if "explain" not in target_method:
        return False
    predict_priority_methods = ["predict_proba", "predict", "predict_log_proba"]
    for src_method in predict_priority_methods:
        src_opts = method_options.get(src_method)
        if src_opts is not None:
            return bool(src_opts.get("case_sensitive", False))
    return False
