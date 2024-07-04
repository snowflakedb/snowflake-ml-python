from absl.testing import absltest

import snowflake.ml._internal.utils.identifier as identifier

SCHEMA_LEVEL_OBJECT_TEST_CASES = [
    ("foo", False, None, None, "foo", ""),
    ("foo/", False, None, None, "foo", "/"),
    ('"foo"', False, None, None, '"foo"', ""),
    ('"foo"/', False, None, None, '"foo"', "/"),
    ("foo/bar", False, None, None, "foo", "/bar"),
    ("foo/bar.gz", False, None, None, "foo", "/bar.gz"),
    ('"foo"/bar.gz', False, None, None, '"foo"', "/bar.gz"),
    ("testschema.foo", False, None, "testschema", "foo", ""),
    ('testschema."foo"', False, None, "testschema", '"foo"', ""),
    ("testschema.foo/bar", False, None, "testschema", "foo", "/bar"),
    ("testschema.foo/bar.gz", False, None, "testschema", "foo", "/bar.gz"),
    ('testschema."foo"/bar.gz', False, None, "testschema", '"foo"', "/bar.gz"),
    ('"testschema".foo', False, None, '"testschema"', "foo", ""),
    ('"testschema"."foo"', False, None, '"testschema"', '"foo"', ""),
    ('"testschema".foo/bar', False, None, '"testschema"', "foo", "/bar"),
    ('"testschema".foo/bar.gz', False, None, '"testschema"', "foo", "/bar.gz"),
    ('"testschema"."foo"/bar.gz', False, None, '"testschema"', '"foo"', "/bar.gz"),
    ("testdb.testschema.foo", True, "testdb", "testschema", "foo", ""),
    ("_testdb.testschema._foo/", False, "_testdb", "testschema", "_foo", "/"),
    ('testdb$."test""s""chema"._f1oo', True, "testdb$", '"test""s""chema"', "_f1oo", ""),
    ("test1db.test$schema.foo1/nytrain/", False, "test1db", "test$schema", "foo1", "/nytrain/"),
    ("test_db.test_schema.foo.nytrain.1.txt", False, "test_db", "test_schema", "foo", ".nytrain.1.txt"),
    ('test_d$b."test.schema".foo$_o/nytrain/', False, "test_d$b", '"test.schema"', "foo$_o", "/nytrain/"),
    (
        '"идентификатор"."test schema"."f.o_o1"',
        True,
        '"идентификатор"',
        '"test schema"',
        '"f.o_o1"',
        "",
    ),
]


class SnowflakeIdentifierTest(absltest.TestCase):
    def test_is_quote_valid(self) -> None:
        self.assertTrue(identifier._is_quoted('"foo"'))
        self.assertTrue(identifier._is_quoted('"""foo"""'))
        self.assertFalse(identifier._is_quoted("FOO"))

    def test_is_quote_invalid(self) -> None:
        with self.assertRaises(ValueError):
            identifier._is_quoted("foo")
        with self.assertRaises(ValueError):
            identifier._is_quoted('foo"')
        with self.assertRaises(ValueError):
            identifier._is_quoted('"bar')
        with self.assertRaises(ValueError):
            identifier._is_quoted('foo"bar')
        with self.assertRaises(ValueError):
            identifier._is_quoted('""foo""')
        with self.assertRaises(ValueError):
            identifier._is_quoted('"foo"""bar"')

    def test_get_unescaped_names(self) -> None:
        self.assertEqual("FOO", identifier.get_unescaped_names("FOO"))
        self.assertEqual("foo", identifier.get_unescaped_names('"foo"'))
        self.assertEqual('"foo"', identifier.get_unescaped_names('"""foo"""'))
        self.assertEqual('foo"bar', identifier.get_unescaped_names('"foo""bar"'))

        input_and_expected_output_tuples = [
            (None, None),
            ("ABC", "ABC"),
            ('"Abc"', "Abc"),
            (["ABC", '"Abc"'], ["ABC", "Abc"]),
        ]

        for input, expected_output in input_and_expected_output_tuples:
            self.assertEqual(identifier.get_unescaped_names(input), expected_output)

    def test_get_inferred_names(self) -> None:
        self.assertEqual("FOO", identifier.get_inferred_names("FOO"))
        self.assertEqual('"foo"', identifier.get_inferred_names("foo"))
        self.assertEqual('"""foo"""', identifier.get_inferred_names('"foo"'))
        self.assertEqual('"foo""bar"', identifier.get_inferred_names('foo"bar'))
        self.assertEqual('"Foo"', identifier.get_inferred_names("Foo"))
        self.assertEqual("FOO1", identifier.get_inferred_names("FOO1"))
        self.assertEqual('"1FOO"', identifier.get_inferred_names("1FOO"))
        self.assertEqual('"FOO 1"', identifier.get_inferred_names("FOO 1"))

        input_and_expected_output_tuples = [
            (None, None),
            ("ABC", "ABC"),
            ("Abc", '"Abc"'),
            ("1COL", '"1COL"'),
            (
                ["ABC", "Abc"],
                ["ABC", '"Abc"'],
            ),
        ]

        for input, expected_output in input_and_expected_output_tuples:
            self.assertEqual(identifier.get_inferred_names(input), expected_output)

    def test_plan_concat(self) -> None:
        """Test vanilla concat with no quotes."""
        self.assertEqual('"demo__task1"', identifier.concat_names(["demo__", "task1"]))
        self.assertEqual('"demo__task1"', identifier.concat_names(["demo", "__", "task1"]))

    def test_user_specified_quotes(self) -> None:
        """Test use of double quote in case of specified quoted ids."""
        self.assertEqual('"demo__task1"', identifier.concat_names(['"demo__"', "task1"]))
        self.assertEqual('"demo__task1"', identifier.concat_names(["demo__", '"task1"']))

    def test_parse_schema_level_object_identifier(self) -> None:
        """Test if the schema level identifiers could be successfully parsed"""

        for test_case in SCHEMA_LEVEL_OBJECT_TEST_CASES:
            with self.subTest():
                self.assertTupleEqual(
                    tuple(test_case[2:]), identifier.parse_schema_level_object_identifier(test_case[0])
                )

    def test_get_schema_level_object_identifier(self) -> None:
        for test_case in SCHEMA_LEVEL_OBJECT_TEST_CASES:
            with self.subTest():
                self.assertEqual(test_case[0], identifier.get_schema_level_object_identifier(*test_case[2:]))

    def test_is_fully_qualified_name(self) -> None:
        for test_case in SCHEMA_LEVEL_OBJECT_TEST_CASES:
            with self.subTest():
                self.assertEqual(test_case[1], identifier.is_fully_qualified_name(test_case[0]))

    def test_resolve_identifier(self) -> None:
        self.assertEqual("FOO", identifier.resolve_identifier("FOO"))
        self.assertEqual("FOO", identifier.resolve_identifier("foo"))
        self.assertEqual("FOO", identifier.resolve_identifier('"FOO"'))
        self.assertEqual('"foo"', identifier.resolve_identifier('"foo"'))
        self.assertEqual('"foo 1"', identifier.resolve_identifier('"foo 1"'))
        self.assertEqual('"""foo"""', identifier.resolve_identifier('"""foo"""'))
        self.assertEqual('"""FOO"""', identifier.resolve_identifier('"""FOO"""'))
        self.assertEqual("FOO", identifier.resolve_identifier("Foo"))
        self.assertEqual("FOO1", identifier.resolve_identifier("FOO1"))
        self.assertEqual("FOO1", identifier.resolve_identifier("Foo1"))

        with self.assertRaises(ValueError):
            identifier.resolve_identifier("1FOO")

        with self.assertRaises(ValueError):
            identifier.resolve_identifier("FOO 1")

        with self.assertRaises(ValueError):
            identifier.resolve_identifier('foo"bar')


if __name__ == "__main__":
    absltest.main()
