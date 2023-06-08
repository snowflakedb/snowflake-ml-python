import collections
import platform
import textwrap
from importlib import metadata as importlib_metadata
from typing import DefaultDict, List, cast

from absl.testing import absltest
from packaging import requirements, specifiers

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import row, session


class EnvUtilsTest(absltest.TestCase):
    def test_validate_pip_requirement_string(self) -> None:
        r = env_utils._validate_pip_requirement_string("python-package==1.0.1")
        self.assertEqual(r.specifier, specifiers.SpecifierSet("==1.0.1"))
        self.assertEqual(r.name, "python-package")

        r = env_utils._validate_pip_requirement_string("python.package==1.0.1")
        self.assertEqual(r.name, "python-package")

        r = env_utils._validate_pip_requirement_string("Python-package==1.0.1")
        self.assertEqual(r.name, "python-package")
        r = env_utils._validate_pip_requirement_string("python-package>=1.0.1,<2,~=1.1,!=1.0.3")
        self.assertEqual(r.specifier, specifiers.SpecifierSet(">=1.0.1, <2, ~=1.1, !=1.0.3"))
        r = env_utils._validate_pip_requirement_string("requests [security,tests] >= 2.8.1, == 2.8.*")
        self.assertSetEqual(r.extras, {"security", "tests"})
        r = env_utils._validate_pip_requirement_string(
            "pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686"
        )
        self.assertEqual(r.name, "pip")
        with self.assertRaises(ValueError):
            env_utils._validate_pip_requirement_string("python==3.8.13")
        with self.assertRaises(ValueError):
            env_utils._validate_pip_requirement_string("python-package=1.0.1")
        with self.assertRaises(ValueError):
            env_utils._validate_pip_requirement_string("_python-package==1.0.1")
        with self.assertRaises(ValueError):
            env_utils._validate_pip_requirement_string('requests; python_version < "2.7"')

    def test_validate_conda_dependency_string(self) -> None:
        c, r = env_utils._validate_conda_dependency_string("python-package==1.0.1")
        self.assertEqual(c, "")
        c, r = env_utils._validate_conda_dependency_string("conda-forge::python-package>=1.0.1,<2,!=1.0.3")
        self.assertEqual(c, "conda-forge")
        self.assertEqual(r.specifier, specifiers.SpecifierSet(">=1.0.1, <2, !=1.0.3"))
        c, r = env_utils._validate_conda_dependency_string("https://repo.anaconda.com/pkgs/snowflake::python-package")
        self.assertEqual(c, "https://repo.anaconda.com/pkgs/snowflake")
        self.assertEqual(r.name, "python-package")

        c, r = env_utils._validate_conda_dependency_string(
            "pip::pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686"
        )
        self.assertEqual(c, "pip")
        self.assertEqual(r.name, "pip")
        self.assertEqual(
            r.url, "https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686"
        )

        c, r = env_utils._validate_conda_dependency_string("pip::requests [security,tests] >= 2.8.1, == 2.8.*")
        self.assertEqual(c, "pip")
        self.assertEqual(r.name, "requests")
        self.assertSetEqual(r.extras, {"security", "tests"})

        with self.assertRaises(ValueError):
            env_utils._validate_conda_dependency_string("python-package=1.0.1")
        with self.assertRaises(ValueError):
            env_utils._validate_conda_dependency_string("python-package~=1.0.1")
        with self.assertRaises(ValueError):
            env_utils._validate_conda_dependency_string("_python-package==1.0.1")
        with self.assertRaises(ValueError):
            env_utils._validate_conda_dependency_string('requests; python_version < "2.7"')
        with self.assertRaises(ValueError):
            env_utils._validate_conda_dependency_string("requests [security,tests] >= 2.8.1, == 2.8.*")
        with self.assertRaises(ValueError):
            env_utils._validate_conda_dependency_string(
                "pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686"
            )

    def test_check_if_requirement_same(self) -> None:
        ra = requirements.Requirement("python-package==1.0.1")
        rb = requirements.Requirement("python-package!=1.0.2")
        self.assertTrue(env_utils._check_if_requirement_same(ra, rb))

        ra = requirements.Requirement("python-package[extra]")
        rb = requirements.Requirement("python-package!=1.0.2")
        self.assertTrue(env_utils._check_if_requirement_same(ra, rb))

        ra = requirements.Requirement("python-package")
        rb = requirements.Requirement("python-package!=1.0.2")
        self.assertTrue(env_utils._check_if_requirement_same(ra, rb))

        ra = requirements.Requirement("python-package")
        rb = requirements.Requirement("another-python-package!=1.0.2")
        self.assertFalse(env_utils._check_if_requirement_same(ra, rb))

        ra = requirements.Requirement("pip >= 23.0.1")
        rb = requirements.Requirement(
            "pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686"
        )
        self.assertTrue(env_utils._check_if_requirement_same(ra, rb))

    def test_append_requirement_list(self) -> None:
        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = [requirements.Requirement("python-package==1.0.1")]
            ra = requirements.Requirement("python-package!=1.0.2")
            trl = [requirements.Requirement("python-package==1.0.1,!=1.0.2")]
            env_utils.append_requirement_list(rl, ra)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = [requirements.Requirement("python-package[extra1]")]
            ra = requirements.Requirement("python-package[extra2]")
            trl = [requirements.Requirement("python-package[extra1, extra2]")]
            env_utils.append_requirement_list(rl, ra)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = [requirements.Requirement("pip==1.0.1")]
            ra = requirements.Requirement(
                "pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686"
            )
            trl = rl + [ra]
            env_utils.append_requirement_list(rl, ra)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = [requirements.Requirement("python-package")]
            ra = requirements.Requirement("python-package")
            trl = rl
            env_utils.append_requirement_list(rl, ra)

        rl = [requirements.Requirement("another-python-package")]
        ra = requirements.Requirement("python-package")
        trl = rl + [ra]
        env_utils.append_requirement_list(rl, ra)
        self.assertListEqual(rl, trl)

    def test_append_conda_dependency(self) -> None:
        rd: DefaultDict[str, List[requirements.Requirement]] = collections.defaultdict(list)
        with self.assertRaises(env_utils.DuplicateDependencyError):
            rd["a"] = [requirements.Requirement("python-package==1.0.1")]
            ra = requirements.Requirement("python-package!=1.0.2")
            trd = {"a": [requirements.Requirement("python-package==1.0.1,!=1.0.2")]}
            env_utils.append_conda_dependency(rd, ("a", ra))

        rd = collections.defaultdict(list)
        rd["a"] = [requirements.Requirement("python-package==1.0.1")]
        ra = requirements.Requirement("another-python-package!=1.0.2")
        trd = {
            "a": [
                requirements.Requirement("python-package==1.0.1"),
                requirements.Requirement("another-python-package!=1.0.2"),
            ]
        }
        env_utils.append_conda_dependency(rd, ("a", ra))
        self.assertDictEqual(rd, trd)

        with self.assertRaises(env_utils.DuplicateDependencyInMultipleChannelsError):
            rd = collections.defaultdict(list)
            rd["b"] = [requirements.Requirement("python-package==1.0.1")]
            ra = requirements.Requirement("python-package!=1.0.2")
            env_utils.append_conda_dependency(rd, ("a", ra))

    def test_validate_pip_requirement_string_list(self) -> None:
        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = ["python-package==1.0.1", "python-package!=1.0.2"]
            env_utils.validate_pip_requirement_string_list(rl)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = ["python-package[extra1]", "python-package[extra2]"]
            env_utils.validate_pip_requirement_string_list(rl)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = [
                "pip==1.0.1",
                "pip @ https://github.com/pypa/pip/archive/1.3.1.zip#sha1=da9234ee9982d4bbb3c72346a6de940a148ea686",
            ]
            env_utils.validate_pip_requirement_string_list(rl)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = ["python-package", "python_package"]
            env_utils.validate_pip_requirement_string_list(rl)

        rl = ["python-package", "another-python-package"]
        trl = [requirements.Requirement("python-package"), requirements.Requirement("another-python-package")]
        self.assertListEqual(env_utils.validate_pip_requirement_string_list(rl), trl)

    def test_validate_conda_dependency_string_list(self) -> None:
        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = ["python-package==1.0.1", "python-package!=1.0.2"]
            env_utils.validate_conda_dependency_string_list(rl)

        with self.assertRaises(env_utils.DuplicateDependencyError):
            rl = ["a::python-package==1.0.1", "a::python-package!=1.0.2"]
            env_utils.validate_conda_dependency_string_list(rl)

        rl = ["python-package==1.0.1", "a::another-python-package!=1.0.2"]
        trd = {
            "a": [requirements.Requirement("another-python-package!=1.0.2")],
            "": [requirements.Requirement("python-package==1.0.1")],
        }
        self.assertDictEqual(env_utils.validate_conda_dependency_string_list(rl), trd)

        with self.assertRaises(env_utils.DuplicateDependencyInMultipleChannelsError):
            rl = ["python-package==1.0.1", "a::python-package!=1.0.2"]
            env_utils.validate_conda_dependency_string_list(rl)

        with self.assertRaises(env_utils.DuplicateDependencyInMultipleChannelsError):
            rl = ["a::python-package==1.0.1", "b::python-package!=1.0.2"]
            env_utils.validate_conda_dependency_string_list(rl)

        with self.assertRaises(env_utils.DuplicateDependencyInMultipleChannelsError):
            rl = ["a::python-package==1.0.1", "python-package!=1.0.2"]
            env_utils.validate_conda_dependency_string_list(rl)

    def test_get_local_installed_version_of_pip_package(self) -> None:
        self.assertEqual(
            requirements.Requirement(f"pip=={importlib_metadata.version('pip')}"),
            env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("pip")),
        )

        self.assertEqual(
            requirements.Requirement(f"pip=={importlib_metadata.version('pip')}"),
            env_utils.get_local_installed_version_of_pip_package(requirements.Requirement("pip>=1.0.0")),
        )

        self.assertEqual(
            requirements.Requirement(f"pip=={importlib_metadata.version('pip')}"),
            env_utils.get_local_installed_version_of_pip_package(
                requirements.Requirement(f"pip=={importlib_metadata.version('pip')}")
            ),
        )

        r = requirements.Requirement(f"pip=={importlib_metadata.version('pip')}")
        self.assertIsNot(
            r,
            env_utils.get_local_installed_version_of_pip_package(r),
        )

        r = requirements.Requirement(f"pip!={importlib_metadata.version('pip')}")
        self.assertIs(
            r,
            env_utils.get_local_installed_version_of_pip_package(r),
        )

        r = requirements.Requirement(env_utils._SNOWML_PKG_NAME)
        self.assertEqual(
            requirements.Requirement(f"{env_utils._SNOWML_PKG_NAME}=={snowml_env.VERSION}"),
            env_utils.get_local_installed_version_of_pip_package(r),
        )

        r = requirements.Requirement("python-package")
        self.assertIs(
            r,
            env_utils.get_local_installed_version_of_pip_package(r),
        )

        with self.assertWarns(UserWarning):
            env_utils.get_local_installed_version_of_pip_package(
                requirements.Requirement(f"pip!={importlib_metadata.version('pip')}")
            )

    def test_relax_requirement_version(self) -> None:
        r = requirements.Requirement("python-package==1.0.1")
        self.assertEqual(env_utils.relax_requirement_version(r), requirements.Requirement("python-package"))

        r = requirements.Requirement("python-package==1.0.1, !=1.0.2")
        self.assertEqual(env_utils.relax_requirement_version(r), requirements.Requirement("python-package"))

        r = requirements.Requirement("python-package[extra]==1.0.1")
        self.assertEqual(env_utils.relax_requirement_version(r), requirements.Requirement("python-package[extra]"))

        r = requirements.Requirement("python-package")
        self.assertEqual(env_utils.relax_requirement_version(r), requirements.Requirement("python-package"))
        self.assertIsNot(env_utils.relax_requirement_version(r), r)

    def test_resolve_conda_environment(self) -> None:
        _SNOWFLAKE_CONDA_CHANNEL_URL = "https://repo.anaconda.com/pkgs/snowflake"
        rl = [requirements.Requirement("numpy")]
        self.assertIsNotNone(
            env_utils.resolve_conda_environment(
                rl, [_SNOWFLAKE_CONDA_CHANNEL_URL], python_version=snowml_env.PYTHON_VERSION
            )
        )

        rl = [requirements.Requirement("numpy==1.22.4")]
        self.assertIsNone(
            env_utils.resolve_conda_environment(
                rl, [_SNOWFLAKE_CONDA_CHANNEL_URL], python_version=snowml_env.PYTHON_VERSION
            )
        )

        rl = [requirements.Requirement(f"numpy=={importlib_metadata.version('numpy')}")]
        self.assertListEqual(
            env_utils.resolve_conda_environment(
                rl,
                ["defaults"],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            [f"numpy=={importlib_metadata.version('numpy')}"],
        )

        rl = [requirements.Requirement(f"numpy<={importlib_metadata.version('numpy')}")]
        self.assertListEqual(
            env_utils.resolve_conda_environment(
                rl,
                ["defaults"],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            [f"numpy=={importlib_metadata.version('numpy')}"],
        )

    def test_validate_requirements_in_snowflake_conda_channel(self) -> None:
        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(
            query=textwrap.dedent(
                """
                SHOW COLUMNS
                LIKE 'runtime_version'
                IN TABLE information_schema.packages;
                """
            ),
            result=mock_data_frame.MockDataFrame(count_result=0),
        )

        query = textwrap.dedent(
            """
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE (package_name = 'pytorch' OR package_name = 'xgboost')
            AND language = 'python';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.3.3"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.5.1"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.7.3"),
            row.Row(PACKAGE_NAME="pytorch", VERSION="1.12.1"),
        ]

        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            sorted(["xgboost", "pytorch"]),
        )

        # Test cache
        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            sorted(["xgboost", "pytorch"]),
        )

        # clear cache
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}

        query = textwrap.dedent(
            """
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE (package_name = 'xgboost')
            AND language = 'python';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.3.3"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.5.1"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.7.3"),
        ]

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            ["xgboost"],
        )

        # Test cache
        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            ["xgboost"],
        )

        query = textwrap.dedent(
            """
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE (package_name = 'pytorch')
            AND language = 'python';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME="pytorch", VERSION="1.12.1"),
        ]

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            sorted(["xgboost", "pytorch"]),
        )

        # Test cache
        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            sorted(["xgboost", "pytorch"]),
        )

        # clear cache
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}

        query = textwrap.dedent(
            """
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE (package_name = 'xgboost')
            AND language = 'python';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.7.0"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.7.1"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.7.3"),
        ]

        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.7.3")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            ["xgboost==1.7.3"],
        )

        # Test cache
        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.7.3")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            ["xgboost==1.7.3"],
        )

        with self.assertRaises(ValueError):
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost<1.7")],
                python_version=snowml_env.PYTHON_VERSION,
            )

        with self.assertRaises(ValueError):
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.7.1, ==1.7.3")],
                python_version=snowml_env.PYTHON_VERSION,
            )

        # clear cache
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}

        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.7.*")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            ["xgboost==1.7.*"],
        )

        # Test cache
        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.7.*")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            ["xgboost==1.7.*"],
        )

        # clear cache
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}

        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertIsNone(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.3.*")],
                python_version=snowml_env.PYTHON_VERSION,
            )
        )

        # Test cache
        self.assertIsNone(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost==1.3.*")],
                python_version=snowml_env.PYTHON_VERSION,
            )
        )

        # clear cache
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}

        query = textwrap.dedent(
            """
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE (package_name = 'python-package')
            AND language = 'python';
            """
        )
        sql_result = [row.Row()]

        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertIsNone(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("python-package")],
                python_version=snowml_env.PYTHON_VERSION,
            )
        )

        env_utils._INFO_SCHEMA_PACKAGES_HAS_RUNTIME_VERSION = None
        m_session = mock_session.MockSession(conn=None, test_case=self)
        m_session.add_mock_sql(
            query=textwrap.dedent(
                """
                SHOW COLUMNS
                LIKE 'runtime_version'
                IN TABLE information_schema.packages;
                """
            ),
            result=mock_data_frame.MockDataFrame(count_result=1),
        )

        query = textwrap.dedent(
            f"""
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE (package_name = 'pytorch' OR package_name = 'xgboost')
            AND language = 'python'
            AND runtime_version = '{platform.python_version_tuple()[0]}.{platform.python_version_tuple()[1]}';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.3.3"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.5.1"),
            row.Row(PACKAGE_NAME="xgboost", VERSION="1.7.3"),
            row.Row(PACKAGE_NAME="pytorch", VERSION="1.12.1"),
        ]

        m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))
        c_session = cast(session.Session, m_session)

        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            sorted(["xgboost", "pytorch"]),
        )

        # Test cache
        self.assertListEqual(
            env_utils.validate_requirements_in_snowflake_conda_channel(
                session=c_session,
                reqs=[requirements.Requirement("xgboost"), requirements.Requirement("pytorch")],
                python_version=snowml_env.PYTHON_VERSION,
            ),
            sorted(["xgboost", "pytorch"]),
        )


if __name__ == "__main__":
    absltest.main()
