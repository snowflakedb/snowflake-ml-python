import tempfile
import textwrap
from importlib import metadata as importlib_metadata
from typing import Dict, List, cast

from absl.testing import absltest
from packaging import requirements

from snowflake.ml._internal import env as snowml_env, env_utils
from snowflake.ml.model import model_signature
from snowflake.ml.model._deploy_client.warehouse import deploy
from snowflake.ml.model._packager.model_meta import model_blob_meta, model_meta
from snowflake.ml.test_utils import exception_utils, mock_data_frame, mock_session
from snowflake.snowpark import row, session

_DUMMY_SIG = {
    "predict": model_signature.ModelSignature(
        inputs=[
            model_signature.FeatureSpec(dtype=model_signature.DataType.FLOAT, name="input"),
        ],
        outputs=[model_signature.FeatureSpec(name="output", dtype=model_signature.DataType.FLOAT)],
    )
}

_DUMMY_BLOB = model_blob_meta.ModelBlobMeta(
    name="model1", model_type="custom", path="mock_path", handler_version="version_0"
)

_BASIC_DEPENDENCIES_FINAL_PACKAGES = list(
    sorted(
        map(
            lambda x: env_utils.get_local_installed_version_of_pip_package(requirements.Requirement(x)),
            model_meta._PACKAGING_CORE_DEPENDENCIES + [env_utils.SNOWPARK_ML_PKG_NAME],
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
                    if basic_dep.name != env_utils.SNOWPARK_ML_PKG_NAME
                },
                env_utils.SNOWPARK_ML_PKG_NAME: [snowml_env.VERSION],
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
        with tempfile.TemporaryDirectory() as tmpdir:
            env_utils._SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE = {}
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir, name="model1", model_type="custom", signatures=_DUMMY_SIG
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                c_session = cast(session.Session, self.m_session)

                final_packages = deploy._get_model_final_packages(meta, c_session)
                self.assertListEqual(final_packages, list(map(str, _BASIC_DEPENDENCIES_FINAL_PACKAGES)))

    def test_get_model_final_packages_no_relax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_utils._SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE = {}
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["pandas==1.0.*"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                c_session = cast(session.Session, self.m_session)
                with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
                    deploy._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_relax(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_utils._SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE = {}
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                c_session = cast(session.Session, self.m_session)

                final_packages = deploy._get_model_final_packages(meta, c_session, relax_version=True)
                self.assertListEqual(
                    final_packages,
                    list(map(str, map(env_utils.relax_requirement_version, _BASIC_DEPENDENCIES_FINAL_PACKAGES))),
                )

    def test_get_model_final_packages_with_pip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_utils._SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE = {}
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                pip_requirements=["python-package"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                c_session = cast(session.Session, self.m_session)
                with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
                    deploy._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_other_channel(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_utils._SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE = {}
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["conda-forge::python_package"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                c_session = cast(session.Session, self.m_session)
                with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
                    deploy._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_non_exist_package(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_utils._SNOWFLAKE_INFO_SCHEMA_PACKAGE_CACHE = {}
            d = {
                **{
                    basic_dep.name: [importlib_metadata.version(basic_dep.name)]
                    for basic_dep in _BASIC_DEPENDENCIES_FINAL_PACKAGES
                    if basic_dep.name != env_utils.SNOWPARK_ML_PKG_NAME
                },
                env_utils.SNOWPARK_ML_PKG_NAME: [snowml_env.VERSION],
            }
            d["python-package"] = []
            self.m_session = mock_session.MockSession(conn=None, test_case=self)
            self.add_packages(d)
            with model_meta.create_model_metadata(
                model_dir_path=tmpdir,
                name="model1",
                model_type="custom",
                signatures=_DUMMY_SIG,
                conda_dependencies=["python-package"],
            ) as meta:
                meta.models["model1"] = _DUMMY_BLOB
                c_session = cast(session.Session, self.m_session)

                with exception_utils.assert_snowml_exceptions(self, expected_original_error_type=RuntimeError):
                    deploy._get_model_final_packages(meta, c_session)


if __name__ == "__main__":
    absltest.main()
