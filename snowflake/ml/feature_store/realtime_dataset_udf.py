"""Warehouse-safe ``map_in_pandas`` wrapper for realtime feature views.

cloudpickle embeds this module into the warehouse UDTF, so it must
have no ``snowflake.ml.*`` imports — the worker doesn't have
``snowflake-ml-python`` installed.
"""

from __future__ import annotations

import textwrap as _textwrap_module
from typing import Any, Callable, Iterator

import pandas as pd

_RTFV_SPINE_ROW_ID_COL = "_RTFV_SPINE_ROW_ID"


def _rehydrate(source: str, name: str, runtime_namespace: dict[str, Any]) -> Callable[..., pd.DataFrame]:
    """Reconstruct ``compute_fn`` by exec-ing its persisted source.

    Args:
        source: Dedented Python source containing ``def <name>(...)``.
        name: ``__name__`` of the function to extract.
        runtime_namespace: Globals dict the source is exec'd in.

    Returns:
        The reconstructed callable.
    """
    namespace: dict[str, Any] = dict(runtime_namespace)
    dedented = _textwrap_module.dedent(source)
    exec(compile(dedented, f"<realtime-config:{name}>", "exec"), namespace)
    return namespace[name]  # type: ignore[no-any-return]


def build_wrapper(
    *,
    compute_fn_source: str,
    compute_fn_name: str,
    runtime_namespace: dict[str, Any],
    request_aliased_cols: list[str],
    upstream_aliased_cols: list[list[str]],
    declared_request_field_names: list[str],
    declared_upstream_feature_groups: list[list[str]],
    rtfv_label: str,
) -> Callable[[Iterator[pd.DataFrame]], Iterator[pd.DataFrame]]:
    """Return a ``map_in_pandas`` wrapper that runs an RTFV's ``compute_fn``.

    The wrapper splits each input batch into per-source pandas frames
    (un-namespacing the SQL-side prefixes), invokes ``compute_fn`` with
    one positional arg per source (RequestSource first, upstreams in
    declaration order), and yields the row id passthrough alongside the
    ``compute_fn`` output columns.

    Args:
        compute_fn_source: Dedented source text of the user's compute_fn.
        compute_fn_name: ``__name__`` of the user's compute_fn.
        runtime_namespace: ``realtime_config._RTFV_RUNTIME_NAMESPACE``.
        request_aliased_cols: Pandas column names for RequestSource fields,
            in declared order. Each is the namespaced alias the SDK emitted
            in the SQL ``SELECT``; the wrapper renames them back to the
            original field names.
        upstream_aliased_cols: Pandas column names per upstream FV source,
            in declared order. Same shape rule as ``request_aliased_cols``.
        declared_request_field_names: Original RequestSource field names
            in declared order.
        declared_upstream_feature_groups: Original feature column names
            per upstream FV in declared order.
        rtfv_label: ``<name>@<version>`` string used in error messages.

    Returns:
        A function suitable for ``Session.map_in_pandas``.
    """

    def _wrapper(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        rehydrated = _rehydrate(compute_fn_source, compute_fn_name, runtime_namespace)

        for pdf in iterator:
            row_ids = pdf[_RTFV_SPINE_ROW_ID_COL].reset_index(drop=True)

            req_renamed = dict(zip(request_aliased_cols, declared_request_field_names))
            request_df = pdf[request_aliased_cols].rename(columns=req_renamed).reset_index(drop=True)
            request_df = request_df[declared_request_field_names]

            upstream_dfs = []
            for aliased, declared_names in zip(upstream_aliased_cols, declared_upstream_feature_groups):
                src_renamed = dict(zip(aliased, declared_names))
                src_df = pdf[aliased].rename(columns=src_renamed).reset_index(drop=True)
                src_df = src_df[declared_names]
                upstream_dfs.append(src_df)

            output_df = rehydrated(request_df, *upstream_dfs)
            if not isinstance(output_df, pd.DataFrame):
                raise TypeError(
                    f"realtime feature view {rtfv_label}: compute_fn must return a "
                    f"pandas DataFrame; got {type(output_df).__name__}."
                )
            output_df = output_df.reset_index(drop=True)

            yield pd.concat([row_ids.rename(_RTFV_SPINE_ROW_ID_COL), output_df], axis=1)

    return _wrapper
