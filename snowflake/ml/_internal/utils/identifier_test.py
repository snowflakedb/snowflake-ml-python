from absl.testing import absltest

import snowflake.ml._internal.utils.identifier as identifier


class SnowflakeIdentifierTest(absltest.TestCase):
    def test_is_quote_valid(self) -> None:
        self.assertTrue(identifier._is_quoted('"foo"'))
        self.assertTrue(identifier._is_quoted('"""foo"""'))
        self.assertFalse(identifier._is_quoted("foo"))

    def test_is_quote_invalid(self) -> None:
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
        self.assertEqual("FOO", identifier.get_unescaped_names("foo"))
        self.assertEqual("foo", identifier.get_unescaped_names('"foo"'))
        self.assertEqual('"foo"', identifier.get_unescaped_names('"""foo"""'))
        self.assertEqual('foo"bar', identifier.get_unescaped_names('"foo""bar"'))

        input_and_expected_output_tuples = [
            (None, None),
            ("Abc", "ABC"),
            ('"Abc"', "Abc"),
            (["Abc", '"Abc"'], ["ABC", "Abc"]),
        ]

        for input, expected_output in input_and_expected_output_tuples:
            self.assertEqual(identifier.get_unescaped_names(input), expected_output)

    def test_plan_concat(self) -> None:
        """Test vanilla concat with no quotes."""
        self.assertEqual("demo__task1", identifier.concat_names(["demo__", "task1"]))

    def test_user_specificed_quotes(self) -> None:
        """Test use of double quote in case of specified quoted ids."""
        self.assertEqual('"demo__task1"', identifier.concat_names(['"demo__"', "task1"]))
        self.assertEqual('"demo__task1"', identifier.concat_names(["demo__", '"task1"']))

    def test_parse_schema_level_object_identifier(self) -> None:
        """Test if the schema level identifiers could be scuuessfully parsed"""
        test_cases = [
            ("testdb.testschema.foo", "testdb", "testschema", "foo", ""),
            ("_testdb.testschema._foo/", "_testdb", "testschema", "_foo", "/"),
            ('testdb$."test""s""chema"._f1oo', "testdb$", '"test""s""chema"', "_f1oo", ""),
            ("test1db.test$schema.foo1/nytrain/", "test1db", "test$schema", "foo1", "/nytrain/"),
            ("test_db.test_schema.foo.nytrain.1.txt", "test_db", "test_schema", "foo", ".nytrain.1.txt"),
            ('test_d$b."test.schema".fo$_o/nytrain/', "test_d$b", '"test.schema"', "fo$_o", "/nytrain/"),
            (
                '"идентификатор"."test schema"."f.o_o1"',
                '"идентификатор"',
                '"test schema"',
                '"f.o_o1"',
                "",
            ),
        ]

        for test_case in test_cases:
            with self.subTest():
                self.assertTupleEqual(
                    tuple(test_case[1:]), identifier.parse_schema_level_object_identifier(test_case[0])
                )


if __name__ == "__main__":
    absltest.main()
