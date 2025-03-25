from absl.testing import absltest

from snowflake import snowpark
from snowflake.ml.jobs import decorators

COMPUTE_POOL = "test_compute_pool"


class JobDecoratorTests(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.compute_pool = COMPUTE_POOL
        cls.session = absltest.mock.MagicMock(spec=snowpark.Session)

    def test_job_decorator_unsupported_arg_type(self) -> None:
        @decorators.remote(self.compute_pool, stage_name="payload_stage", session=self.session)  # type: ignore[misc]
        def decojob_fn2(a, b: int, session) -> None:  # type: ignore[no-untyped-def]
            pass

        with self.assertRaisesRegex(ValueError, "Unable to serialize positional arg 2.*'Session'"):
            decojob_fn2(1, 2, self.session)

        with self.assertRaisesRegex(ValueError, "Unable to serialize keyword arg 'session'.*'Session'"):
            decojob_fn2(1, 2, session=self.session)

        with self.assertRaisesRegex(ValueError, "Unable to serialize positional arg 0.*'Session'"):
            decojob_fn2(self.session, 2, self.session)


if __name__ == "__main__":
    absltest.main()
