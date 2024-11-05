import inspect
import itertools
import os
import tempfile
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import cloudpickle
from absl.testing import absltest, parameterized
from packaging import requirements, specifiers
from typing_extensions import Concatenate, ParamSpec

from snowflake.ml._internal import env, env_utils, file_utils
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


def get_modified_test_cases(
    test_cases: Union[List[Dict[str, Any]], List[Tuple[Any, ...]]],
    additional_cases: List[Any],
    additional_case_arg_name: str,
    naming_type: object,
) -> Union[List[Dict[str, Any]], List[Tuple[Any, ...]]]:
    if all(isinstance(tc, dict) for tc in test_cases):
        modified_test_cases = [{**t1, additional_case_arg_name: t2} for t1 in test_cases for t2 in additional_cases]
        if naming_type is parameterized._NAMED:
            for tc in modified_test_cases:
                tc["testcase_name"] += f"_{tc[additional_case_arg_name]}"
    else:
        raise ValueError("Unsupported parameterized test argument input.")
    return modified_test_cases


class CommonTestBase(parameterized.TestCase):
    def setUp(self) -> None:
        """Creates Snowpark and Snowflake environments for testing."""
        self.session = test_env_utils.get_available_session()

    def tearDown(self) -> None:
        if not snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
            self.session.close()

    @classmethod
    def sproc_test(
        kclass: Type[_V],
        local: bool = True,
        test_callers_rights: bool = True,
        test_owners_rights: bool = True,
        additional_packages: Optional[List[str]] = None,
    ) -> Callable[
        [Callable[Concatenate[_V, _T_args], None]],
        Union[parameterized._ParameterizedTestIter, Callable[Concatenate[_V, _T_args], None]],
    ]:
        def decorator(
            fn: Union[parameterized._ParameterizedTestIter, Callable[Concatenate[_V, _T_args], None]]
        ) -> Union[parameterized._ParameterizedTestIter, Callable[Concatenate[_V, _T_args], None]]:
            if snowpark_utils.is_in_stored_procedure():  # type: ignore[no-untyped-call]
                return fn

            if isinstance(fn, parameterized._ParameterizedTestIter):
                actual_method = fn._test_method
                original_name = fn._original_name
                naming_type = fn._naming_type
                test_cases = list(fn.testcases)
                if naming_type is parameterized._NAMED:
                    method_list = [method.__name__ for method in fn]
                else:
                    method_list = [f"{original_name}{i}" for i, _ in enumerate(fn)]
            else:
                actual_method = fn
                original_name = fn.__name__
                naming_type = parameterized._ARGUMENT_REPR
                test_cases = [{}]
                method_list = [original_name]

            assert len(method_list) > 0
            test_module = inspect.getmodule(actual_method)
            assert test_module
            assert test_module.__file__
            test_module_path = os.path.abspath(test_module.__file__)
            ind = test_module_path.rfind(f"tests{os.sep}")
            assert ind > 0
            rel_path = test_module_path[ind:]
            rel_path = os.path.splitext(rel_path)[0]
            test_module_name = rel_path.replace(os.sep, ".")

            def test_wrapper(
                self: _V,
                /,
                *args: _T_args.args,
                _sproc_test_mode: Literal["local", "owner", "caller"],
                **kwargs: _T_args.kwargs,
            ) -> None:
                if _sproc_test_mode == "local":
                    actual_method(self, *args, **kwargs)
                    return

                def _in_sproc_test(execute_as: Literal["owner", "caller"] = "owner") -> None:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        snowml_zip_module_filename = os.path.join(tmpdir, "snowflake-ml-python.zip")
                        file_utils.zip_python_package(snowml_zip_module_filename, "snowflake.ml")

                        tests_zip_module_filename = os.path.join(tmpdir, "snowflake-ml-test.zip")
                        file_utils.zip_python_package(tests_zip_module_filename, "tests")

                        imports = [snowml_zip_module_filename, tests_zip_module_filename]
                        packages = additional_packages or []
                        for req_str in _snowml_requirements.ALL_REQUIREMENTS:
                            req = requirements.Requirement(req_str)
                            # Remove "_" not in req once Snowpark 1.11.0 available, it is a workaround for their bug.
                            if any(offending in req.name for offending in ["snowflake-connector-python", "pyarrow"]):
                                continue
                            # != and ~= are not supported by Snowpark
                            req.specifier = specifiers.SpecifierSet(
                                ",".join(
                                    [
                                        str(spec)
                                        for spec in req.specifier
                                        if spec.operator in ["<", "<=", ">", ">=", "=="]
                                    ]
                                )
                            )
                            packages.append(str(req))

                        cloudpickle.register_pickle_by_value(test_module)

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
                            import fcntl
                            import os
                            import sys
                            import threading
                            import unittest
                            import zipfile
                            from types import TracebackType
                            from typing import Optional, Type

                            class FileLock:
                                def __enter__(self) -> None:
                                    self._lock = threading.Lock()
                                    self._lock.acquire()
                                    self._fd = open("/tmp/lockfile.LOCK", "w+")
                                    fcntl.lockf(self._fd, fcntl.LOCK_EX)

                                def __exit__(
                                    self,
                                    exc_type: Optional[Type[BaseException]],
                                    exc: Optional[BaseException],
                                    traceback: Optional[TracebackType],
                                ) -> None:
                                    self._fd.close()
                                    self._lock.release()

                            IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
                            import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]

                            for file_name in ["snowflake-ml-python", "snowflake-ml-test"]:
                                zip_path = os.path.join(import_dir, file_name + ".zip")
                                for sys_path in sys.path:
                                    if file_name + ".zip" in sys_path:
                                        sys.path.remove(sys_path)

                                extracted_dir_path = os.path.join("/tmp", file_name)

                                with FileLock():
                                    if not os.path.isdir(extracted_dir_path):
                                        with zipfile.ZipFile(zip_path, "r") as myzip:
                                            myzip.extractall(extracted_dir_path)
                                        sys.path.append(extracted_dir_path)

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

                        for method_name in method_list:
                            test_in_sproc(
                                self.session, f"{test_module_name}.{self.__class__.__qualname__}.{method_name}"
                            )

                        cloudpickle.unregister_pickle_by_value(test_module)

                _in_sproc_test(execute_as=_sproc_test_mode)

            additional_cases = []
            if local:
                additional_cases.append("local")

            if test_callers_rights:
                additional_cases.append("caller")

            if test_owners_rights:
                additional_cases.append("owner")

            modified_test_cases = get_modified_test_cases(test_cases, additional_cases, "_sproc_test_mode", naming_type)

            return parameterized._ParameterizedTestIter(
                test_wrapper, modified_test_cases, naming_type, original_name=original_name
            )

        return decorator

    @classmethod
    def compatibility_test(
        kclass: Type[_V],
        prepare_fn_factory: Callable[[_V], Tuple[Callable[[session.Session, _R_args], None], _R_args]],
        version_range: Optional[str] = None,
        additional_packages: Optional[List[str]] = None,
    ) -> Callable[
        [Union[parameterized._ParameterizedTestIter, Callable[Concatenate[_V, _T_args], None]]],
        parameterized._ParameterizedTestIter,
    ]:
        def decorator(
            fn: Union[parameterized._ParameterizedTestIter, Callable[Concatenate[_V, _T_args], None]]
        ) -> parameterized._ParameterizedTestIter:
            if isinstance(fn, parameterized._ParameterizedTestIter):
                actual_method = fn._test_method
                original_name = fn._original_name
                naming_type = fn._naming_type
                test_cases = list(fn.testcases)
            else:
                actual_method = fn
                original_name = fn.__name__
                naming_type = parameterized._ARGUMENT_REPR
                test_cases = [{}]

            def test_wrapper(self: _V, /, *args: _T_args.args, _snowml_pkg_ver: str, **kwargs: _T_args.kwargs) -> None:
                prepare_fn, prepare_fn_args = prepare_fn_factory(self)
                packages = additional_packages or []

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

                final_packages = packages[:] + [f"snowflake-ml-python=={_snowml_pkg_ver}", "snowflake-snowpark-python"]

                with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as temp_file:
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
                        execute_as="caller",
                    )

                prepare_fn_sproc(*prepare_fn_args, session=self.session)

                actual_method(self, *args, **kwargs)

            additional_cases = [
                str(pkg_ver)
                for pkg_ver in env_utils.get_matched_package_versions_in_information_schema(
                    test_env_utils.get_available_session(),
                    [requirements.Requirement(f"{env_utils.SNOWPARK_ML_PKG_NAME}{version_range}")],
                    python_version=env.PYTHON_VERSION,
                )[env_utils.SNOWPARK_ML_PKG_NAME]
            ]

            modified_test_cases = get_modified_test_cases(test_cases, additional_cases, "_snowml_pkg_ver", naming_type)

            return parameterized._ParameterizedTestIter(
                test_wrapper, modified_test_cases, naming_type, original_name=original_name
            )

        return decorator


if __name__ == "__main__":
    absltest.main()
