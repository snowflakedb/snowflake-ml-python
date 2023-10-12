import functools
import inspect
import os
import tempfile
from typing import Any, Callable, Type, TypeVar

import cloudpickle
from absl.testing import absltest, parameterized

from snowflake.ml._internal import file_utils
from snowflake.ml.utils import connection_params
from snowflake.snowpark import functions as F, session
from snowflake.snowpark._internal import utils as snowpark_utils
from tests.integ.snowflake.ml.test_utils import _snowml_requirements

T = TypeVar("T")


class CommonTestBase(parameterized.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self.session = (
            session.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
            if not snowpark_utils.is_in_stored_procedure()
            else session._get_active_session()
        )

    def tearDown(self) -> None:
        if not snowpark_utils.is_in_stored_procedure():
            self.session.close()

    @classmethod
    def sproc_test(
        kclass: Type["CommonTestBase"], local: bool = True
    ) -> Callable[[Callable[["CommonTestBase", T], None]], Callable[["CommonTestBase", T], None]]:
        def decorator(fn: Callable[["CommonTestBase", T], None]) -> Callable[["CommonTestBase", T], None]:
            @functools.wraps(fn)
            def test_wrapper(self: "CommonTestBase", *args: Any, **kwargs: Any) -> None:
                if snowpark_utils.is_in_stored_procedure():
                    fn(self, *args, **kwargs)
                    return

                if local:
                    fn(self, *args, **kwargs)

                test_module = inspect.getmodule(fn)
                assert test_module
                cloudpickle.register_pickle_by_value(test_module)
                assert test_module.__file__
                test_module_path = os.path.abspath(test_module.__file__)
                ind = test_module_path.rfind(f"tests{os.sep}")
                assert ind > 0
                rel_path = test_module_path[ind:]
                rel_path = os.path.splitext(rel_path)[0]
                test_module_name = rel_path.replace(os.sep, ".")
                test_name = f"{test_module_name}.{fn.__qualname__}"

                with tempfile.TemporaryDirectory() as tmpdir:
                    snowml_path, snowml_start_path = file_utils.get_package_path("snowflake.ml")

                    snowml_zip_module_filename = os.path.join(tmpdir, "snowflake-ml-python.zip")
                    with file_utils.zip_file_or_directory_to_stream(snowml_path, snowml_start_path) as input_stream:
                        with open(snowml_zip_module_filename, "wb") as f:
                            f.write(input_stream.getbuffer())

                    tests_path, tests_start_path = file_utils.get_package_path("tests")

                    tests_zip_module_filename = os.path.join(tmpdir, "snowflake-ml-test.zip")
                    with file_utils.zip_file_or_directory_to_stream(tests_path, tests_start_path) as input_stream:
                        with open(tests_zip_module_filename, "wb") as f:
                            f.write(input_stream.getbuffer())

                    imports = [snowml_zip_module_filename, tests_zip_module_filename]
                    packages = [
                        req for req in _snowml_requirements.REQUIREMENTS if "snowflake-connector-python" not in req
                    ]

                    @F.sproc(
                        is_permanent=False,
                        packages=packages,
                        replace=True,
                        session=self.session,
                        anonymous=True,
                        imports=imports,
                    )
                    def test_in_sproc(sess: session.Session, test_name: str) -> None:
                        import unittest

                        loader = unittest.TestLoader()

                        suite = loader.loadTestsFromName(test_name)
                        result = unittest.TextTestRunner(verbosity=2, failfast=False).run(suite)
                        if len(result.errors) > 0 or len(result.failures) > 0:
                            raise RuntimeError(
                                "Unit test failed unexpectedly with at least one error. "
                                f"Errors: {result.errors} Failures: {result.failures}"
                            )
                        if result.testsRun == 0:
                            raise RuntimeError("Unit test does not run any test.")

                    test_in_sproc(self.session, test_name)

            return test_wrapper

        return decorator


if __name__ == "__main__":
    absltest.main()
