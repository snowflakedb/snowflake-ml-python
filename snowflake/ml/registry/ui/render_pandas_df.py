import functools
import urllib
from dataclasses import dataclass
from typing import Callable, Dict, Hashable, Optional, cast

import pandas as pd

from snowflake.ml._internal.utils import identifier
from snowflake.ml.registry import model_registry
from snowflake.snowpark import functions
from snowflake.snowpark._internal import type_utils

_COLUMN_ORDER = [
    "NAME",
    "ID",
    "VERSION",
    "TYPE",
    "DESCRIPTION",
]

_COLUMNS_TO_REMOVE_QUOTE = [
    "NAME",
    "ID",
    "CREATION_ROLE",
    "ROLE",
]

_MULTILINE_COLUMNS = ["CREATION_ENVIRONMENT_SPEC", "TAGS", "VALUE[ATTRIBUTE_NAME]"]
_HISTORY_COLUMNS = ["EVENT_TIMESTAMP", "ROLE", "ATTRIBUTE_NAME", "VALUE[ATTRIBUTE_NAME]"]
_RENAME_COLUMNS = {"NAME": "Model Name", "IDS": "Number of Models", "VALUE[ATTRIBUTE_NAME]": "ATTRIBUTE_VALUE"}


def _encode_url(key: str, value: str, params: Dict[str, str]) -> str:
    """Returns encoded URL given the params.

    Args:
        key: Key of URL parameter for current entry.
        value: Value of URL parameter. This is also used as anchor text.
        params: Other existing URL parameters.

    Returns:
        Encoded URL.
    """
    value = identifier.remove_quote_if_quoted(value)
    params[key] = value
    return f"<a href='/?{urllib.parse.urlencode(params)}' target='_self''>{value}</a>"


def _pandas_df_formatters(params: Dict[str, str]) -> Dict[str, Callable[[str], str]]:
    fmt: Dict[str, Callable[[str], str]] = {}
    for c in _COLUMNS_TO_REMOVE_QUOTE:
        fmt[c] = functools.partial(identifier.remove_quote_if_quoted)
    for c in _MULTILINE_COLUMNS:
        fmt[c] = lambda x: x.replace("\n", "<br>")

    fmt["NAME"] = lambda v: _encode_url("model", v, params)
    fmt["ID"] = lambda v: _encode_url("id", v, params)
    return fmt


def _format_pandas_df(df: pd.DataFrame, params: Dict[str, str], transpose: bool = False) -> str:
    """Format & render given pandas DataFrame."""
    cols = set(df.columns)
    # Fix the column ordering
    new_order = []
    for c in _COLUMN_ORDER:
        if c in cols:
            new_order.append(c)
    for c in df.columns.to_list():
        if c not in _COLUMN_ORDER:
            new_order.append(c)
    df = df[new_order]

    formatters = _pandas_df_formatters(params)
    # Rename columns, if needed
    columns = {}
    for old, new in _RENAME_COLUMNS.items():
        if old in cols:
            columns[old] = new
            if old in formatters:
                formatters[new] = formatters[old]
                del formatters[old]
    df.rename(columns=columns, inplace=True)
    cols = set(df.columns)

    # Fix the formatting
    if transpose:
        for c, f in formatters.items():
            if c in cols:
                df[c] = df[c].apply(f)
        df = df.transpose()
    return str(
        # Cast is adding here since both Mapping and Callable type are invariant.
        df.to_html(
            escape=False,
            formatters=None if transpose else cast(Dict[Hashable, Callable[[object], str]], formatters),
            header=False if transpose else True,
        )
    )


@dataclass
class RenderedContent:
    """Class holding elements of all the rendered contents."""

    header: Optional[str] = None
    description: Optional[str] = None
    history_header: Optional[str] = "Events"
    history: Optional[str] = None
    found: bool = True


class DfRenderer:
    """Renderer dealing with all pandas DataFrame."""

    def __init__(self, *, registry: model_registry.ModelRegistry, url_params: Dict[str, str]) -> None:
        """Initializes renderer.

        Args:
            registry: Must be a valid registry object.
            url_params: Optional url parameters. In absence of any, it renders the root page.
        """
        assert registry
        self._registry = registry
        self._url_params = {}
        for k, v in url_params.items():
            self._url_params[k] = identifier.remove_quote_if_quoted(v)

    def render(self) -> RenderedContent:
        """Returns high level description as HTML content.

        Returns:
            RenderedContent with appropriate fields filled, if succesful; error (str) Otherwise.
        """
        model_list = self._registry.list_models()
        history = None
        transpose = False
        content = RenderedContent()
        if "id" in self._url_params:
            model_id = self._url_params["id"]
            model_list = model_list.filter(model_list["ID"] == model_id)
            transpose = True
            content.header = "Details of model ID: " + model_id
            history = (
                self._registry.get_model_history(id=model_id)
                .select(_HISTORY_COLUMNS)
                .order_by("EVENT_TIMESTAMP", ascending=False)
            )
        elif "model" in self._url_params:
            model_name = self._url_params["model"]
            model_list = (
                model_list.filter(model_list["NAME"] == model_name)
                .select("ID", "CREATION_TIME", "TAGS")
                .order_by("CREATION_TIME", ascending=False)
            )
            content.header = "List of models with model name: " + model_name
            content.history_header = "Last 20 events"
            history = (
                self._registry.get_history()
                .select(_HISTORY_COLUMNS)
                .order_by("EVENT_TIMESTAMP", ascending=False)
                .limit(20)
            )
        else:
            model_list = model_list.group_by("NAME").agg(functions.count(type_utils.ColumnOrName("ID")).alias("IDS"))
            content.header = "List of registered models"

        model_list_pd = model_list.to_pandas()
        if model_list_pd.size:
            content.description = _format_pandas_df(model_list_pd, self._url_params, transpose=transpose)
        else:
            content.found = False

        if history:
            history_pd = history.to_pandas()
            content.history = _format_pandas_df(history_pd, self._url_params)

        return content
