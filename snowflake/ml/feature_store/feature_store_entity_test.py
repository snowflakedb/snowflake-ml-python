"""Unit tests for SQL-string-literal escaping in FeatureStore DDL.

Originally driven by a SQL injection report against ``register_entity`` where a
``desc`` containing ``'`` terminated the ``COMMENT = '...'`` literal and let the
caller append arbitrary DDL. The same raw-interpolation pattern existed in six
sister sites (``update_entity`` + five FeatureView ``COMMENT='...'`` paths), so
these tests pin the fix across every site that interpolates user-controlled
``desc`` into a single-quoted SQL string literal.

Each test asserts the SQL string that would be sent to Snowflake — either the
captured argument to ``session.sql(...)`` or the return value of a pure
string-builder helper — has its embedded ``'`` doubled (``''``), the Snowflake
DDL convention for string-literal escaping.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from absl.testing import absltest

from snowflake.ml._internal.utils.sql_identifier import SqlIdentifier
from snowflake.ml.feature_store import feature_store as fs_mod
from snowflake.ml.feature_store.entity import Entity
from snowflake.ml.feature_store.feature_store import FeatureStore
from snowflake.ml.feature_store.feature_view import (
    FeatureView,
    FeatureViewStatus,
    StorageConfig,
    StorageFormat,
)
from snowflake.snowpark.types import StringType, StructField, StructType

# A representative exploit payload from the original report. The single quote
# right after ``x]`` would terminate ``COMMENT = '...'`` in the unescaped code
# and let the trailing tokens execute as a separate statement.
_EXPLOIT_DESC = "x]' ->> create user hackersqli password = 'aaa' --"


def _new_fs_with_mocks(*, session: MagicMock | None = None) -> FeatureStore:
    """Construct a bare-bones FeatureStore with all I/O mocked.

    Mirrors the helper in ``feature_store_feature_group_test.py`` so these
    tests don't need a live Snowpark session or warehouse. ``get_current_warehouse``
    returns ``None`` so the ``@dispatch_decorator`` warehouse switch is a no-op
    (otherwise it would try ``SqlIdentifier(<MagicMock>)`` and crash).
    """
    fs = object.__new__(FeatureStore)
    sess = session or MagicMock()
    sess.get_current_warehouse.return_value = None
    object.__setattr__(fs, "_session", sess)
    object.__setattr__(
        fs,
        "_config",
        fs_mod._FeatureStoreConfig(database=SqlIdentifier("DB"), schema=SqlIdentifier("SCH")),
    )
    object.__setattr__(fs, "_telemetry_stmp", {})
    object.__setattr__(fs, "_metadata_manager", MagicMock())
    object.__setattr__(fs, "_default_warehouse", SqlIdentifier("WH"))
    return fs


def _make_feature_view_mock(
    *,
    desc: str,
    is_tiled: bool = False,
    storage_config: StorageConfig | None = None,
    status: FeatureViewStatus = FeatureViewStatus.ACTIVE,
    refresh_freq: str | None = "1 minute",
) -> MagicMock:
    """Build a ``MagicMock`` standing in for a ``FeatureView`` for string-builder tests.

    The DDL string builders read a handful of attributes; we configure just
    those rather than constructing a real ``FeatureView`` (which would require
    a Snowpark DataFrame).
    """
    fv = MagicMock(spec=FeatureView)
    fv.desc = desc
    fv.is_tiled = is_tiled
    fv.query = "SELECT 1 AS A FROM T"
    fv.storage_config = storage_config
    fv.refresh_freq = refresh_freq
    fv.refresh_mode = "AUTO"
    fv.initialize = "ON_CREATE"
    fv.cluster_by = None
    fv.status = status
    fv.warehouse = SqlIdentifier("WH")
    fv.fully_qualified_name = MagicMock(return_value='"DB"."SCH"."FV$V1"')
    return fv


def _captured_sql(session: MagicMock) -> list[str]:
    return [call.args[0] for call in session.sql.call_args_list]


def _assert_safely_quoted(test_case: absltest.TestCase, sql: str, desc: str, *, clause_prefix: str) -> None:
    """Assert ``desc`` appears in ``sql`` only as a properly-doubled string literal.

    ``clause_prefix`` is the exact run of SQL immediately preceding the opening
    quote of the literal we want to inspect — for example ``"COMMENT = "`` or
    ``"SET COMMENT = "``. We anchor on that to find the right literal in
    multi-statement output.
    """
    anchor = clause_prefix + "'"
    start = sql.find(anchor)
    test_case.assertGreaterEqual(start, 0, f"could not find {anchor!r} in:\n{sql}")
    literal_start = start + len(clause_prefix) + 1  # past the opening quote
    # The closing quote is the first ``'`` that is not part of a doubled ``''``.
    cursor = literal_start
    while cursor < len(sql):
        if sql[cursor] != "'":
            cursor += 1
            continue
        if cursor + 1 < len(sql) and sql[cursor + 1] == "'":
            cursor += 2
            continue
        break
    test_case.assertLess(cursor, len(sql), f"unterminated SQL literal in:\n{sql}")
    literal_body = sql[literal_start:cursor]
    expected_body = desc.replace("'", "''")
    test_case.assertEqual(
        literal_body,
        expected_body,
        f"literal body did not match doubled-quote expectation. " f"clause_prefix={clause_prefix!r}, desc={desc!r}",
    )


class RegisterEntityDescEscapingTest(absltest.TestCase):
    """Regression for the original report: ``register_entity`` desc must not break out of its literal."""

    def _new_fs(self) -> tuple[FeatureStore, MagicMock]:
        sess = MagicMock()
        fs = _new_fs_with_mocks(session=sess)
        object.__setattr__(fs, "_find_object", MagicMock(return_value=[]))
        object.__setattr__(fs, "get_entity", MagicMock(return_value=Entity("FOO", ["id"])))
        return fs, sess

    def test_exploit_desc_does_not_terminate_comment_literal(self) -> None:
        fs, sess = self._new_fs()
        evil = Entity("FOO", ["id"], desc=_EXPLOIT_DESC)

        fs.register_entity(evil)

        create_sqls = [s for s in _captured_sql(sess) if "CREATE TAG" in s]
        self.assertEqual(len(create_sqls), 1, _captured_sql(sess))
        _assert_safely_quoted(self, create_sqls[0], _EXPLOIT_DESC, clause_prefix="COMMENT = ")

    def test_apostrophe_in_desc_is_doubled(self) -> None:
        fs, sess = self._new_fs()
        fs.register_entity(Entity("FOO", ["id"], desc="It's a customer entity"))

        create_sqls = [s for s in _captured_sql(sess) if "CREATE TAG" in s]
        self.assertEqual(len(create_sqls), 1)
        self.assertIn("COMMENT = 'It''s a customer entity'", create_sqls[0])

    def test_empty_desc_still_emits_empty_literal(self) -> None:
        fs, sess = self._new_fs()
        fs.register_entity(Entity("FOO", ["id"]))

        create_sqls = [s for s in _captured_sql(sess) if "CREATE TAG" in s]
        self.assertEqual(len(create_sqls), 1)
        self.assertIn("COMMENT = ''", create_sqls[0])


class UpdateEntityDescEscapingTest(absltest.TestCase):
    """``update_entity`` runs ALTER TAG with the same literal shape — same fix applies."""

    def _new_fs(self) -> tuple[FeatureStore, MagicMock]:
        sess = MagicMock()
        fs = _new_fs_with_mocks(session=sess)
        # The found_rows path runs through list_entities().filter(...).collect(...).
        # The exact filter result content doesn't matter; len() > 0 is enough.
        list_entities_df = MagicMock()
        list_entities_df.filter.return_value.collect.return_value = [{"DESC": "old"}]
        object.__setattr__(fs, "list_entities", MagicMock(return_value=list_entities_df))
        object.__setattr__(fs, "get_entity", MagicMock(return_value=Entity("FOO", ["id"])))
        return fs, sess

    def test_exploit_desc_does_not_terminate_comment_literal(self) -> None:
        fs, sess = self._new_fs()

        fs.update_entity("FOO", desc=_EXPLOIT_DESC)

        alter_sqls = [s for s in _captured_sql(sess) if "ALTER TAG" in s]
        self.assertEqual(len(alter_sqls), 1, _captured_sql(sess))
        _assert_safely_quoted(self, alter_sqls[0], _EXPLOIT_DESC, clause_prefix="SET COMMENT = ")


class EntityDetailLookupEscapingTest(absltest.TestCase):
    """Tag lookups escape ``'`` and ``\\`` in the object name passed to ``ENTITY_DETAIL``."""

    _DATABASE = "\"DB\\')||(SELECT CURRENT_ROLE())||('\""

    def setUp(self) -> None:
        super().setUp()
        self.sess = MagicMock()
        self.fs = _new_fs_with_mocks(session=self.sess)
        object.__setattr__(
            self.fs,
            "_config",
            fs_mod._FeatureStoreConfig(database=SqlIdentifier(self._DATABASE), schema=SqlIdentifier("SCH")),
        )

    def test_lookup_tags_escapes_object_name(self) -> None:
        self.sess.sql.return_value.collect.return_value = [['{"tagReferencesInfo": {"tagReferenceList": []}}']]

        self.fs._lookup_tags("table", "FV$V1")

        self.sess.sql.assert_called_once()
        self.assertIn(
            r"""'"DB\\'')||(SELECT CURRENT_ROLE())||(''".SCH.FV$V1'""",
            self.sess.sql.call_args.args[0],
        )

    def test_lookup_tagged_objects_escapes_tag_name(self) -> None:
        self.sess.sql.return_value.collect.return_value = [['{"referencedEntities": {"tagReferenceList": []}}']]

        self.fs._lookup_tagged_objects("SNOWML_FEATURE_STORE_OBJECT")

        self.sess.sql.assert_called_once()
        self.assertIn(
            r"""'"DB\\'')||(SELECT CURRENT_ROLE())||(''".SCH.SNOWML_FEATURE_STORE_OBJECT'""",
            self.sess.sql.call_args.args[0],
        )


class BuildOfflineUpdateQueriesDescEscapingTest(absltest.TestCase):
    """``_build_offline_update_queries`` covers two sites: STATIC ALTER VIEW and DT alter_dt."""

    def test_static_branch_escapes_desc(self) -> None:
        fs = _new_fs_with_mocks()
        fv = _make_feature_view_mock(desc="ignored", status=FeatureViewStatus.STATIC, refresh_freq=None)

        ops, _rollback = fs._build_offline_update_queries(
            feature_view=fv,
            refresh_freq=None,
            warehouse=None,
            initialization_warehouse=fs_mod._KEEP_CURRENT,
            desc=_EXPLOIT_DESC,
        )

        self.assertEqual(len(ops), 1)
        _, sql = ops[0]
        self.assertIn("ALTER VIEW", sql)
        _assert_safely_quoted(self, sql, _EXPLOIT_DESC, clause_prefix="SET COMMENT = ")

    def test_dt_branch_escapes_desc(self) -> None:
        fs = _new_fs_with_mocks()
        # ACTIVE DT-backed FV with a duration refresh_freq exercises alter_dt.
        fv = _make_feature_view_mock(desc=_EXPLOIT_DESC, refresh_freq="1 minute")

        ops, rollback = fs._build_offline_update_queries(
            feature_view=fv,
            refresh_freq=None,
            warehouse=None,
            initialization_warehouse=fs_mod._KEEP_CURRENT,
            desc=_EXPLOIT_DESC,
        )

        alter_dt_sqls = [sql for _op, sql in ops + rollback if "ALTER DYNAMIC TABLE" in sql]
        self.assertGreaterEqual(len(alter_dt_sqls), 1)
        for sql in alter_dt_sqls:
            _assert_safely_quoted(self, sql, _EXPLOIT_DESC, clause_prefix="COMMENT = ")


class CreateDynamicTableQueryDescEscapingTest(absltest.TestCase):
    """``_create_dynamic_table_query`` has two desc interpolations (iceberg and snowflake formats)."""

    def _call(self, *, storage_config: StorageConfig | None) -> str:
        fs = _new_fs_with_mocks()
        fv = _make_feature_view_mock(desc=_EXPLOIT_DESC, storage_config=storage_config)
        return fs._create_dynamic_table_query(
            override_clause="",
            table_name='"DB"."SCH"."FV$V1"',
            column_descs="",
            schedule_task=False,
            feature_view=fv,
            tagging_clause="",
            warehouse="WH",
        )

    def test_snowflake_format_escapes_desc(self) -> None:
        sql = self._call(storage_config=None)
        _assert_safely_quoted(self, sql, _EXPLOIT_DESC, clause_prefix="COMMENT = ")

    def test_iceberg_format_escapes_desc(self) -> None:
        sql = self._call(
            storage_config=StorageConfig(
                format=StorageFormat.ICEBERG,
                external_volume="MY_VOLUME",
                base_location="feature_store/fv",
            )
        )
        _assert_safely_quoted(self, sql, _EXPLOIT_DESC, clause_prefix="COMMENT = ")


class CreateOfflineFeatureViewViewQueryDescEscapingTest(absltest.TestCase):
    """``_create_offline_feature_view_view_query`` emits a CREATE VIEW with COMMENT = '...'."""

    def test_exploit_desc_does_not_terminate_comment_literal(self) -> None:
        fs = _new_fs_with_mocks()
        fv = _make_feature_view_mock(desc=_EXPLOIT_DESC)

        sql = fs._create_offline_feature_view_view_query(
            overwrite_clause="",
            view_name='"DB"."SCH"."FV$V1"',
            column_descs='"A"',
            feature_view=fv,
            tagging_clause_str="",
        )

        _assert_safely_quoted(self, sql, _EXPLOIT_DESC, clause_prefix="COMMENT = ")


class CreateOfflineFeatureViewViewQueryColumnListTest(absltest.TestCase):
    """``_create_offline_feature_view_view_query`` column-list rendering."""

    def test_empty_column_descs_omits_column_list(self) -> None:
        """When ``_build_column_descs`` returns ``""`` (``feature_descs is None``),
        the VIEW DDL must not contain an empty ``()`` column list, which Snowflake
        rejects."""
        fs = _new_fs_with_mocks()
        fv = _make_feature_view_mock(desc="")

        sql = fs._create_offline_feature_view_view_query(
            overwrite_clause=" OR REPLACE",
            view_name='"DB"."SCH"."FV$V1"',
            column_descs="",
            feature_view=fv,
            tagging_clause_str="",
        )

        self.assertIn('CREATE OR REPLACE VIEW "DB"."SCH"."FV$V1"', sql)
        self.assertNotRegex(sql, r"VIEW\s+\S+\s*\(\s*\)")
        self.assertRegex(sql, r'VIEW\s+"DB"\."SCH"\."FV\$V1"\s+COMMENT')

    def test_non_empty_column_descs_includes_column_list(self) -> None:
        """A non-empty ``column_descs`` clause is emitted inside a ``(...)`` column list."""
        fs = _new_fs_with_mocks()
        fv = _make_feature_view_mock(desc="")

        sql = fs._create_offline_feature_view_view_query(
            overwrite_clause=" OR REPLACE",
            view_name='"DB"."SCH"."FV$V1"',
            column_descs="user_id COMMENT 'join key', score COMMENT 'feature'",
            feature_view=fv,
            tagging_clause_str="",
        )

        self.assertIn("(user_id COMMENT 'join key', score COMMENT 'feature')", sql)


class CreateRollupFeatureViewDescEscapingTest(absltest.TestCase):
    """The rollup DT path interpolates ``feature_view.desc`` in the same way.

    We reach the desc literal via the same pure builder; rather than exercise
    the full ``_create_rollup_feature_view`` flow (which needs a real parent
    feature view, mapping DataFrame, and join-key plumbing), we assert the
    same site by reading the source DDL string. This is the one site that
    isn't extractable as a standalone helper, so we keep the assertion
    narrow: just confirm the fixed form is present in the file.
    """

    def test_rollup_dt_desc_uses_quote_helper(self) -> None:
        import inspect

        src = inspect.getsource(FeatureStore._create_rollup_feature_view)
        # After the fix, the rollup DT's COMMENT clause routes through the
        # helper instead of interpolating ``feature_view.desc`` raw.
        self.assertNotIn("COMMENT = '{feature_view.desc}'", src)
        self.assertIn("sql_literal.string_literal(feature_view.desc)", src)


class ReadFeatureViewQueryEscapingTest(absltest.TestCase):
    """``read_feature_view`` puts ``keys`` values in string literals and ``feature_names`` in the SELECT list."""

    def setUp(self) -> None:
        super().setUp()
        self.fs = _new_fs_with_mocks()
        self.fv = MagicMock(spec=FeatureView)
        self.fv.entities = [Entity(name="E", join_keys=["ID"])]
        self.fv.output_schema = StructType([StructField("ID", StringType()), StructField('"f1"', StringType())])

    def test_backslash_in_key_value_is_escaped(self) -> None:
        where_clause, _ = self.fs._build_where_clause_for_keys(self.fv, [["x\\"], [") OR 1=1 --"]])

        self.assertEqual(where_clause, r""" WHERE ("ID" = 'x\\') OR ("ID" = ') OR 1=1 --')""")

    def test_case_sensitive_feature_name_is_not_requoted(self) -> None:
        select_clause = self.fs._build_select_clause_and_validate(self.fv, ['"f1"'])

        self.assertEqual(select_clause, '"ID", "f1"')


if __name__ == "__main__":
    absltest.main()
