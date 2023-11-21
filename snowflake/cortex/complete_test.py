import _test_util
from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import _complete
from snowflake.snowpark import functions, types


class CompleteTest(absltest.TestCase):
    model = "|model|"
    prompt = "|prompt|"

    @staticmethod
    def complete_for_test(model: str, prompt: str) -> str:
        return f"answered: {model}, {prompt}"

    def setUp(self) -> None:
        self._session = _test_util.create_test_session()
        functions.udf(
            self.complete_for_test,
            name="complete",
            return_type=types.StringType(),
            input_types=[types.StringType(), types.StringType()],
            is_permanent=False,
        )

    def tearDown(self) -> None:
        self._session.sql("drop function complete(string,string)").collect()
        self._session.close()

    def test_complete_str(self) -> None:
        res = _complete._complete_impl("complete", self.model, self.prompt)
        self.assertEqual(self.complete_for_test(self.model, self.prompt), res)

    def test_complete_column(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model=self.model, prompt=self.prompt)])
        df_out = df_in.select(_complete._complete_impl("complete", functions.col("model"), functions.col("prompt")))
        res = df_out.collect()[0][0]
        self.assertEqual(self.complete_for_test(self.model, self.prompt), res)


if __name__ == "__main__":
    absltest.main()
