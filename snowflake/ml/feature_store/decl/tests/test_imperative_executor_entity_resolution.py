"""Tests for entity resolution when FV ordered_entity_column_names use join keys."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from snowflake.ml._internal.exceptions import (
    error_codes,
    exceptions as snowml_exceptions,
)
from snowflake.ml.feature_store.decl.imperative_executor import (
    _get_entity_for_feature_view_ref,
)
from snowflake.ml.test_utils import pytest_driver


def test_get_entity_direct_hit_when_name_equals_ref() -> None:
    fs = MagicMock()
    ent = MagicMock(name="Entity")
    ent.name = "USER_ID"
    fs.get_entity.return_value = ent
    assert _get_entity_for_feature_view_ref(fs, "USER_ID") is ent
    fs.get_entity.assert_called_once_with("USER_ID")
    fs.list_entities.assert_not_called()


def test_get_entity_join_key_fallback_when_entity_name_differs() -> None:
    fs = MagicMock()
    not_found = snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_FOUND,
        original_exception=ValueError("Cannot find Entity with name: USER_ID."),
    )

    def _get(name: str) -> Any:
        if name == "USER_ID":
            raise not_found
        if name == "USER_BATCH_DECL":
            ent = MagicMock(name="ResolvedEntity")
            ent.name = "USER_BATCH_DECL"
            return ent
        raise AssertionError(f"unexpected get_entity({name!r})")

    fs.get_entity.side_effect = _get

    df = MagicMock()
    df.collect.return_value = [{"NAME": "USER_BATCH_DECL", "JOIN_KEYS": '["USER_ID"]'}]
    fs.list_entities.return_value = df

    out = _get_entity_for_feature_view_ref(fs, "USER_ID")
    assert out.name == "USER_BATCH_DECL"
    fs.list_entities.assert_called_once()


def test_get_entity_join_key_ambiguous_raises() -> None:
    fs = MagicMock()
    not_found = snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_FOUND,
        original_exception=ValueError("missing"),
    )
    fs.get_entity.side_effect = not_found
    df = MagicMock()
    df.collect.return_value = [
        {"NAME": "E1", "JOIN_KEYS": '["USER_ID"]'},
        {"NAME": "E2", "JOIN_KEYS": '["USER_ID"]'},
    ]
    fs.list_entities.return_value = df
    with pytest.raises(ValueError, match="Ambiguous entity resolution"):
        _get_entity_for_feature_view_ref(fs, "USER_ID")


def _fs_with_not_found_then_rows(rows: list[dict[str, Any]], resolved_name: str) -> Any:
    """A FeatureStore mock whose direct get_entity misses, then resolves by join key."""
    fs = MagicMock()
    not_found = snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_FOUND,
        original_exception=ValueError("Cannot find Entity."),
    )

    def _get(name: str) -> Any:
        if name == resolved_name:
            ent = MagicMock(name="ResolvedEntity")
            ent.name = resolved_name
            return ent
        raise not_found

    fs.get_entity.side_effect = _get
    df = MagicMock()
    df.collect.return_value = rows
    fs.list_entities.return_value = df
    return fs


def test_quoted_case_sensitive_ref_matches_bare_deployed_key() -> None:
    # The exact exported shape: entities carries the re-quoted ``"userId"`` and
    # the deployed entity declares the bare stored join key ``userId``.
    fs = _fs_with_not_found_then_rows(
        [{"NAME": "USERID", "JOIN_KEYS": '["userId"]'}],
        resolved_name="USERID",
    )
    out = _get_entity_for_feature_view_ref(fs, '"userId"')
    assert out.name == "USERID"


def test_quoted_all_upper_ref_still_resolves() -> None:
    fs = _fs_with_not_found_then_rows(
        [{"NAME": "USER_ID", "JOIN_KEYS": '["USER_ID"]'}],
        resolved_name="USER_ID",
    )
    out = _get_entity_for_feature_view_ref(fs, '"USER_ID"')
    assert out.name == "USER_ID"


def test_join_keys_as_python_list_resolves() -> None:
    fs = _fs_with_not_found_then_rows(
        [{"NAME": "USERID", "JOIN_KEYS": ["userId"]}],
        resolved_name="USERID",
    )
    out = _get_entity_for_feature_view_ref(fs, '"userId"')
    assert out.name == "USERID"


def test_join_keys_as_comma_separated_string_resolves() -> None:
    # Legacy SHOW TAGS surface: a bare comma-joined cell, not JSON.
    fs = _fs_with_not_found_then_rows(
        [{"NAME": "MULTI", "JOIN_KEYS": "USER_ID,SESSION_ID"}],
        resolved_name="MULTI",
    )
    out = _get_entity_for_feature_view_ref(fs, "SESSION_ID")
    assert out.name == "MULTI"


def test_join_keys_json_scalar_does_not_match_characterwise() -> None:
    # A JSON scalar (not a list) must not be iterated character by character
    # into the comparison set — ``"U"`` / ``"S"`` etc. must not match.
    fs = MagicMock()
    not_found = snowml_exceptions.SnowflakeMLException(
        error_code=error_codes.NOT_FOUND,
        original_exception=ValueError("missing"),
    )
    fs.get_entity.side_effect = not_found
    df = MagicMock()
    df.collect.return_value = [{"NAME": "E1", "JOIN_KEYS": '"USER_ID"'}]
    fs.list_entities.return_value = df
    with pytest.raises(ValueError, match="no entity declares join key"):
        _get_entity_for_feature_view_ref(fs, "U")


def test_multi_key_entity_resolves_each_quoted_ref_to_same_entity() -> None:
    # A single entity declares two case-sensitive join keys; each quoted ref
    # resolves to it, so _build_feature_view dedups them to one Entity.
    rows = [{"NAME": "SESSION_USER", "JOIN_KEYS": '["userId", "sessionId"]'}]
    fs = _fs_with_not_found_then_rows(rows, resolved_name="SESSION_USER")
    assert _get_entity_for_feature_view_ref(fs, '"userId"').name == "SESSION_USER"
    assert _get_entity_for_feature_view_ref(fs, '"sessionId"').name == "SESSION_USER"


def test_empty_after_unquote_ref_raises_empty_entry() -> None:
    fs = MagicMock()
    with pytest.raises(ValueError, match="entry is empty"):
        _get_entity_for_feature_view_ref(fs, '""')
    fs.get_entity.assert_not_called()


def test_embedded_quote_key_resolves() -> None:
    fs = _fs_with_not_found_then_rows(
        [{"NAME": "WEIRD", "JOIN_KEYS": ['user"Id']}],
        resolved_name="WEIRD",
    )
    out = _get_entity_for_feature_view_ref(fs, '"user""Id"')
    assert out.name == "WEIRD"


if __name__ == "__main__":
    pytest_driver.main()
