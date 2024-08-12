from types import GeneratorType

from absl.testing import absltest

from snowflake import snowpark
from snowflake.cortex import Complete, CompleteOptions
from snowflake.ml.utils import connection_params
from snowflake.snowpark import Session, functions

_OPTIONS = CompleteOptions(  # random params
    max_tokens=10,
    temperature=0.7,
    top_p=1,
)
_MODEL_NAME = "mistral-7b"
_PROMPT = "How are you feeling today?"
_CONVERSATION_HISTORY_PROMPT = [
    {"role": "system", "content": "Reply to every message in an unnecessarily adversarial tone."},
    {"role": "user", "content": "How are you feeling today?"},
]


class CompleteSQLTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_snowpark_mode(self) -> None:
        df_in = self._session.create_dataframe([snowpark.Row(model=_MODEL_NAME, prompt=_PROMPT)])
        df_out = df_in.select(Complete(functions.col("model"), functions.col("prompt")))
        res = df_out.collect()[0][0]
        self.assertIsInstance(res, str)
        self.assertTrue(res)

    def test_snowpark_mode_conversation_history(self) -> None:
        df_in = self._session.create_dataframe(
            [snowpark.Row(model=_MODEL_NAME, prompt=_CONVERSATION_HISTORY_PROMPT, options=_OPTIONS)]
        )
        df_out = df_in.select(
            Complete(functions.col("model"), functions.col("prompt"), options=functions.col("options"))
        )
        res = df_out.collect()[0][0]
        self.assertIsInstance(res, str)
        self.assertTrue(res)

    def test_immediate_mode_conversation_history(self) -> None:
        res = Complete(
            model=_MODEL_NAME,
            prompt=_CONVERSATION_HISTORY_PROMPT,
            options=_OPTIONS,
            session=self._session,
        )
        self.assertIsInstance(res, str)
        self.assertTrue(res)

    def test_immediate_mode_no_options(self) -> None:
        res = Complete(
            model=_MODEL_NAME,
            prompt=_PROMPT,
            session=self._session,
        )
        self.assertIsInstance(res, str)
        self.assertTrue(res)

    def test_immediate_mode_populated_options(self) -> None:
        res = Complete(
            model=_MODEL_NAME,
            prompt=_PROMPT,
            options=_OPTIONS,
            session=self._session,
        )
        self.assertIsInstance(res, str)
        self.assertTrue(res)

    def test_immediate_mode_empty_options(self) -> None:
        res = Complete(
            model=_MODEL_NAME,
            prompt=_PROMPT,
            options={},
            session=self._session,
        )
        self.assertIsInstance(res, str)
        self.assertTrue(res)


class CompleteRestTest(absltest.TestCase):
    def setUp(self) -> None:
        self._session = Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()

    def tearDown(self) -> None:
        self._session.close()

    def test_streaming(self) -> None:
        result = Complete(
            model=_MODEL_NAME,
            prompt=_PROMPT,
            session=self._session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        for out in result:
            self.assertIsInstance(out, str)
            self.assertTrue(out)  # nonempty

    def test_streaming_conversation_history(self) -> None:
        result = Complete(
            model=_MODEL_NAME,
            prompt=_CONVERSATION_HISTORY_PROMPT,
            session=self._session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        for out in result:
            self.assertIsInstance(out, str)
            self.assertTrue(out)  # nonempty

    def test_streaming_with_options(self) -> None:
        result = Complete(
            model=_MODEL_NAME,
            prompt=_PROMPT,
            options=_OPTIONS,
            session=self._session,
            stream=True,
        )
        self.assertIsInstance(result, GeneratorType)
        for out in result:
            self.assertIsInstance(out, str)
            self.assertTrue(out)  # nonempty


if __name__ == "__main__":
    absltest.main()
