import textwrap
from importlib import metadata as importlib_metadata
from typing import Dict, List, cast

from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import _model_meta, model_signature
from snowflake.ml.model._deploy_client.warehouse import deploy
from snowflake.ml.test_utils import mock_data_frame, mock_session
from snowflake.snowpark import row, session

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}

_BASIC_DEPENDENCIES_FINAL_PACKAGES = list(
    sorted(
        map(
            lambda x: env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x)),
            _model_meta._BASIC_DEPENDENCIES + [env_utils._SNOWML_PKG_NAME],
        ),
        key=lambda x: x.name,
    )
)


class TestFinalPackagesWithoutConda(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        env_utils._INFO_SCHEMA_PACKAGES_HAS_RUNTIME_VERSION = None
        cls.m_session = mock_session.MockSession(conn=None, test_case=None)
        cls.m_session.add_mock_sql(
            query=textwrap.dedent(
                """
                SHOW COLUMNS
                LIKE 'runtime_version'
                IN TABLE information_schema.packages;
                """
            ),
            result=mock_data_frame.MockDataFrame(count_result=0),
        )

    def setUp(self) -> None:
        self.add_packages(
            {
                **{
                    basic_dep.name: [importlib_metadata.version(basic_dep.name)]
                    for basic_dep in _BASIC_DEPENDENCIES_FINAL_PACKAGES
                    if basic_dep.name != env_utils._SNOWML_PKG_NAME
                },
                env_utils._SNOWML_PKG_NAME: [snowml_env.VERSION],
            }
        )

    @classmethod
    def tearDownClass(cls) -> None:
        pass

    def add_packages(self, packages_dicts: Dict[str, List[str]]) -> None:
        pkg_names_str = " OR ".join(f"package_name = '{pkg}'" for pkg in sorted(packages_dicts.keys()))
        query = textwrap.dedent(
            f"""
            SELECT PACKAGE_NAME, VERSION
            FROM information_schema.packages
            WHERE ({pkg_names_str})
            AND language = 'python';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME=pkg, VERSION=pkg_ver)
            for pkg, pkg_vers in packages_dicts.items()
            for pkg_ver in pkg_vers
        ]
        if len(sql_result) == 0:
            sql_result = [row.Row()]

        self.m_session.add_mock_sql(query=query, result=mock_data_frame.MockDataFrame(sql_result))

    def test_get_model_final_packages(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(name="model1", model_type="custom", signatures=_DUMMY_SIG)
        c_session = cast(session.Session, self.m_session)

        final_packages = deploy._get_model_final_packages(meta, c_session)
        self.assertListEqual(final_packages, list(map(str, _BASIC_DEPENDENCIES_FINAL_PACKAGES)))

    def test_get_model_final_packages_no_relax(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas==1.0.*"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            deploy._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_relax(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas==1.0.*"]
        )
        c_session = cast(session.Session, self.m_session)

        final_packages = deploy._get_model_final_packages(meta, c_session, relax_version=True)
        self.assertListEqual(final_packages, sorted(list(map(lambda x: x.name, _BASIC_DEPENDENCIES_FINAL_PACKAGES))))

    def test_get_model_final_packages_with_pip(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, pip_requirements=["python-package"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            deploy._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_other_channel(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1",
            model_type="custom",
            signatures=_DUMMY_SIG,
            conda_dependencies=["conda-forge::python_package"],
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            deploy._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_non_exist_package(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        d = {
            **{
                basic_dep.name: [importlib_metadata.version(basic_dep.name)]
                for basic_dep in _BASIC_DEPENDENCIES_FINAL_PACKAGES
                if basic_dep.name != env_utils._SNOWML_PKG_NAME
            },
            env_utils._SNOWML_PKG_NAME: [snowml_env.VERSION],
        }
        d["python-package"] = []
        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.add_packages(d)
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["python-package"]
        )
        c_session = cast(session.Session, self.m_session)

        with self.assertRaises(RuntimeError):
            deploy._get_model_final_packages(meta, c_session)


if __name__ == "__main__":
    absltest.main()
