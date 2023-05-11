import sys
import textwrap
from importlib import metadata as importlib_metadata
from typing import Dict, List, cast

from absl.testing import absltest

from snowflake.ml._internal import env_utils
from snowflake.ml.model import _model_meta, _udf_util, model_signature
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
    sorted(map(lambda x: f"{x}=={importlib_metadata.version(x)}", _model_meta._BASIC_DEPENDENCIES))
)


class TestFinalPackagesWithoutConda(absltest.TestCase):
    def setUp(self) -> None:
        self._temp_conda = None
        if sys.modules.get("conda"):
            self._temp_conda = sys.modules["conda"]
        sys.modules["conda"] = None  # type: ignore[assignment]

        self.m_session = mock_session.MockSession(conn=None, test_case=self)
        self.add_packages(
            {basic_dep: [importlib_metadata.version(basic_dep)] for basic_dep in _model_meta._BASIC_DEPENDENCIES}
        )

    def tearDown(self) -> None:
        if self._temp_conda:
            sys.modules["conda"] = self._temp_conda
        else:
            del sys.modules["conda"]

    def add_packages(self, packages_dicts: Dict[str, List[str]]) -> None:
        pkg_names_str = " OR ".join(f"package_name = '{pkg}'" for pkg in sorted(packages_dicts.keys()))
        query = textwrap.dedent(
            f"""
            SELECT *
            FROM information_schema.packages
            WHERE ({pkg_names_str})
            AND language = 'python';
            """
        )
        sql_result = [
            row.Row(PACKAGE_NAME=pkg, VERSION=pkg_ver, LANGUAGE="python")
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
        with self.assertWarns(RuntimeWarning):
            final_packages = _udf_util._get_model_final_packages(meta, c_session)
            self.assertListEqual(final_packages, _BASIC_DEPENDENCIES_FINAL_PACKAGES)

    def test_get_model_final_packages_no_relax(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas<1"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertWarns(RuntimeWarning):
            with self.assertRaises(RuntimeError):
                _udf_util._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_relax(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas<1"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertWarns(RuntimeWarning):
            final_packages = _udf_util._get_model_final_packages(meta, c_session, relax_version=True)
            self.assertListEqual(final_packages, _BASIC_DEPENDENCIES_FINAL_PACKAGES)

    def test_get_model_final_packages_with_pip(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, pip_requirements=["python-package"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            _udf_util._get_model_final_packages(meta, c_session)

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
            _udf_util._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_non_exist_package(self) -> None:
        env_utils._SNOWFLAKE_CONDA_PACKAGE_CACHE = {}
        d = {basic_dep: [importlib_metadata.version(basic_dep)] for basic_dep in _model_meta._BASIC_DEPENDENCIES}
        d["python-package"] = []
        self.add_packages(d)
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["python-package"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertWarns(RuntimeWarning):
            with self.assertRaises(RuntimeError):
                _udf_util._get_model_final_packages(meta, c_session)


class TestFinalPackagesWithConda(absltest.TestCase):
    def setUp(self) -> None:
        self.m_session = mock_session.MockSession(conn=None, test_case=self)

    def tearDown(self) -> None:
        pass

    def test_get_model_final_packages(self) -> None:
        meta = _model_meta.ModelMetadata(name="model1", model_type="custom", signatures=_DUMMY_SIG)
        c_session = cast(session.Session, self.m_session)
        final_packages = _udf_util._get_model_final_packages(meta, c_session, relax_version=True)
        self.assertIsNotNone(final_packages)

    def test_get_model_final_packages_no_relax(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas<1"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            _udf_util._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_relax(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["pandas<1"]
        )
        c_session = cast(session.Session, self.m_session)
        final_packages = _udf_util._get_model_final_packages(meta, c_session, relax_version=True)
        self.assertIsNotNone(final_packages)

    def test_get_model_final_packages_with_pip(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, pip_requirements=["python_package"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            _udf_util._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_other_channel(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1",
            model_type="custom",
            signatures=_DUMMY_SIG,
            conda_dependencies=["conda-forge::python_package"],
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            _udf_util._get_model_final_packages(meta, c_session)

    def test_get_model_final_packages_with_non_exist_package(self) -> None:
        meta = _model_meta.ModelMetadata(
            name="model1", model_type="custom", signatures=_DUMMY_SIG, conda_dependencies=["python_package"]
        )
        c_session = cast(session.Session, self.m_session)
        with self.assertRaises(RuntimeError):
            _udf_util._get_model_final_packages(meta, c_session)


if __name__ == "__main__":
    absltest.main()
