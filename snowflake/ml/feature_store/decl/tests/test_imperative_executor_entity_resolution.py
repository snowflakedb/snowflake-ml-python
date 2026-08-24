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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
