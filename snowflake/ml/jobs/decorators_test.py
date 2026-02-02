from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml.jobs import decorators

COMPUTE_POOL = "test_compute_pool"
DATABASE = "MOCK_DB"
SCHEMA = "MOCK_schema"


class JobDecoratorTests(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute_pool = COMPUTE_POOL
        cls.session = absltest.mock.MagicMock(spec=snowpark.Session)
        cls.session.get_current_database.return_value = DATABASE
        cls.session.get_current_schema.return_value = SCHEMA

    def test_job_decorator_unsupported_arg_type(self) -> None:
        @decorators.remote(self.compute_pool, stage_name="payload_stage", session=self.session)
        def decojob_fn2(a, b: int, mysession, testsession) -> None:  # type: ignore[no-untyped-def]
            pass

        with self.assertRaisesRegex(
            TypeError, "Expected only one Session-type argument, but got both mysession and testsession."
        ):
            decojob_fn2(1, 2, self.session, self.session)

        with self.assertRaisesRegex(
            TypeError, "Expected only one Session-type argument, but got both mysession and testsession."
        ):
            decojob_fn2(1, 2, self.session, testsession=self.session)


if __name__ == "__main__":
    absltest.main()
