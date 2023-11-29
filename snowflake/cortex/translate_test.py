import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _translate
from snowflake.snowpark import functions, types


class TranslateTest(absltest.TestCase):
    text = "|text|"
    from_language = "|from_language|"
    to_language = "|to_language|"

    @staticmethod
    def translate_for_test(text: str, from_language: str, to_language: str) -> str:
        return f"translated: {text}, {from_language}, {to_language}"

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.translate_for_test,
            name="translate_for_test",
            return_type=types.StringType(),
            input_types=[types.StringType(), types.StringType(), types.StringType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function translate_for_test(string,string,string)").collect()
        self._session.close()

    def test_translate_str(self) -> None:
        res = _translate._translate_impl(
            "translate_for_test",
            self.text,
            self.from_language,
            self.to_language,
        )
        self.assertEqual(self.translate_for_test(self.text, self.from_language, self.to_language), res)

    def test_translate_column(self) -> None:
        df_in = self._session.create_dataframe(
            [snowpark.Row(text=self.text, from_language=self.from_language, to_language=self.to_language)]
        )
        df_out = df_in.select(
            _translate._translate_impl(
                "translate_for_test",
                functions.col("text"),
                functions.col("from_language"),
                functions.col("to_language"),
            )
        )
        res = df_out.collect()[0][0]
        self.assertEqual(self.translate_for_test(self.text, self.from_language, self.to_language), res)


if __name__ == "__main__":
    absltest.main()
