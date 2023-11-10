import functools
import inspect
import itertools
import os
import tempfile
from typing import Any, Callable, List, Literal, Optional, Tuple, Type, TypeVar

import cloudpickle
from absl.testing import absltest, parameterized
from typing_extensions import Concatenate, ParamSpec

from snowflake.ml._internal import file_utils
from snowflake.ml.utils import connection_params
from snowflake.snowpark import functions as F, session
from snowflake.snowpark._internal import udf_utils, utils as snowpark_utils
from tests.integ.snowflake.ml.test_utils import _snowml_requirements, test_env_utils

_V = TypeVar("_V", bound="CommonTestBase")
_T_args = ParamSpec("_T_args")
_R_args = TypeVar("_R_args")


def get_function_body(func: Callable[..., Any]) -> str:
    source_lines = inspect.getsourcelines(func)[0]
    source_lines_generator = itertools.dropwhile(lambda x: x.startswith("@"), source_lines)
    first_line: str = next(source_lines_generator)
    indentation = len(first_line) - len(first_line.lstrip())
    first_line = first_line.strip()
    if not first_line.startswith("def "):
        return first_line.rsplit(":")[-1].strip()
    elif not first_line.endswith(":"):
        for line in source_lines_generator:
            line = line.strip()
            if line.endswith(":"):
                break
    # Find the indentation of the first line
    return "".join([line[indentation:] for line in source_lines_generator])


class CommonTestBase(parameterized.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self.session = (
            session.Session.builder.configs(connection_params.SnowflakeLoginOptions()).create()
            if not snowpark_utils.is_in_stored_procedure()  # type: ignore[no-untyped-call] #
            else session._get_active_session()
        )

    def tearDown(self) -> None:
        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            self.session.close()

    @classmethod
    def sproc_test(
        kclass: Type[_V], local: bool = True, test_callers_rights: bool = True
    ) -> Callable[[Callable[Concatenate[_V, _T_args], None]], Callable[Concatenate[_V, _T_args], None]]:
        def decorator(fn: Callable[Concatenate[_V, _T_args], None]) -> Callable[Concatenate[_V, _T_args], None]:
            @functools.wraps(fn)
            def test_wrapper(self: _V, /, *args: _T_args.args, **kwargs: _T_args.kwargs) -> None:
                if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
                    fn(self, *args, **kwargs)
                    return

                if local:
                    with self.subTest("Local Test"):
                        fn(self, *args, **kwargs)

                def _in_sproc_test(execute_as: Literal["owner", "caller"] = "owner") -> None:
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
                            req
                            for req in _snowml_requirements.REQUIREMENTS
                            # Remove "_" not in req once Snowpark 1.11.0 available, it is a workaround for their bug.
                            if "snowflake-connector-python" not in req and "_" not in req
                        ]

                        @F.sproc(  # type: ignore[misc]
                            is_permanent=False,
                            packages=packages,  # type: ignore[arg-type]
                            replace=True,
                            session=self.session,
                            anonymous=(execute_as == "caller"),
                            imports=imports,  # type: ignore[arg-type]
                            execute_as=execute_as,
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

                        cloudpickle.unregister_pickle_by_value(test_module)

                with self.subTest("In-sproc Test (Owner's rights)"):
                    _in_sproc_test(execute_as="owner")

                if test_callers_rights:
                    with self.subTest("In-sproc Test (Caller's rights)"):
                        _in_sproc_test(execute_as="caller")

            return test_wrapper

        return decorator

    @classmethod
    def compatibility_test(
        kclass: Type[_V],
        prepare_fn_factory: Callable[[_V], Tuple[Callable[[session.Session, _R_args], None], _R_args]],
        version_range: Optional[str] = None,
        additional_packages: Optional[List[str]] = None,
    ) -> Callable[[Callable[Concatenate[_V, _T_args], None]], Callable[Concatenate[_V, _T_args], None]]:
        def decorator(fn: Callable[Concatenate[_V, _T_args], None]) -> Callable[Concatenate[_V, _T_args], None]:
            @functools.wraps(fn)
            def test_wrapper(self: _V, /, *args: _T_args.args, **kwargs: _T_args.kwargs) -> None:
                prepare_fn, prepare_fn_args = prepare_fn_factory(self)
                if additional_packages:
                    packages = additional_packages
                else:
                    packages = []

                _, _, return_type, input_types = udf_utils.extract_return_input_types(
                    prepare_fn, return_type=None, input_types=None, object_type=snowpark_utils.TempObjectType.PROCEDURE
                )

                func_body = get_function_body(prepare_fn)
                func_params = inspect.signature(prepare_fn).parameters
                func_name = prepare_fn.__name__

                seen_first_arg = False
                first_arg_name = None
                arg_list = []
                for arg_name in func_params.keys():
                    if not seen_first_arg:
                        seen_first_arg = True
                        first_arg_name = arg_name
                    else:
                        arg_list.append(arg_name)

                assert first_arg_name is not None, "The prepare function must have at least one argument"
                func_source = f"""
import snowflake.snowpark

def {func_name}({first_arg_name}: snowflake.snowpark.Session, {", ".join(arg_list)}):
{func_body}
"""

                for pkg_ver in test_env_utils.get_package_versions_in_server(
                    self.session, f"snowflake-ml-python{version_range}"
                ):
                    with self.subTest(f"Testing with snowflake-ml-python version {pkg_ver}"):
                        final_packages = packages[:] + [f"snowflake-ml-python=={pkg_ver}"]

                        with tempfile.NamedTemporaryFile(
                            "w", encoding="utf-8", suffix=".py", delete=False
                        ) as temp_file:
                            temp_file.write(func_source)
                            temp_file.flush()

                            # Instead of using decorator, we register from file to prevent pickling anything from
                            # current env.
                            prepare_fn_sproc = self.session.sproc.register_from_file(
                                file_path=temp_file.name,
                                func_name=func_name,
                                return_type=return_type,
                                input_types=input_types,
                                is_permanent=False,
                                packages=final_packages,
                                replace=True,
                            )

                        prepare_fn_sproc(*prepare_fn_args, session=self.session)

                        fn(self, *args, **kwargs)

            return test_wrapper

        return decorator


if __name__ == "__main__":
    absltest.main()
