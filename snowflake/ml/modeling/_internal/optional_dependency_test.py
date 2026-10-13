from typing import Callable
from unittest import mock

from absl.testing import absltest

from snowflake.ml.modeling._internal import optional_dependency


def _fake_is_installed(installed: set[str]) -> Callable[[str], bool]:
    def _is_installed(import_name: str) -> bool:
        return import_name in installed

    return _is_installed


class OptionalDependencyTest(absltest.TestCase):
    def _check(self, package_name: str, installed: set[str]) -> Exception | None:
        with mock.patch.object(optional_dependency, "_is_installed", _fake_is_installed(installed)):
            try:
                optional_dependency.check_modeling_dependencies(package_name)
            except ImportError as exc:
                return exc
        return None

    def test_required_import_names(self) -> None:
        self.assertEqual(optional_dependency._required_import_names("snowflake.ml.modeling.linear_model"), ["sklearn"])
        self.assertEqual(optional_dependency._required_import_names("snowflake.ml.modeling.preprocessing"), ["sklearn"])
        self.assertEqual(
            optional_dependency._required_import_names("snowflake.ml.modeling.xgboost"), ["sklearn", "xgboost"]
        )
        self.assertEqual(
            optional_dependency._required_import_names("snowflake.ml.modeling.lightgbm"), ["sklearn", "lightgbm"]
        )

    def test_missing_sklearn_raises_actionable_error(self) -> None:
        exc = self._check("snowflake.ml.modeling.linear_model", installed=set())
        assert exc is not None
        self.assertEqual(
            str(exc),
            "'snowflake.ml.modeling.linear_model' requires scikit-learn, which is not installed. "
            "As of snowflake-ml-python 2.0, scikit-learn is an optional dependency of "
            "snowflake-ml-python that is no longer installed automatically. Install the missing "
            "package in your environment to continue using snowflake.ml.modeling.",
        )

    def test_xgboost_subpackage_missing_only_xgboost(self) -> None:
        exc = self._check("snowflake.ml.modeling.xgboost", installed={"sklearn"})
        assert exc is not None
        # Only the missing package is named; scikit-learn is present.
        self.assertEqual(
            str(exc),
            "'snowflake.ml.modeling.xgboost' requires xgboost, which is not installed. "
            "As of snowflake-ml-python 2.0, xgboost is an optional dependency of "
            "snowflake-ml-python that is no longer installed automatically. Install the missing "
            "package in your environment to continue using snowflake.ml.modeling.",
        )

    def test_xgboost_subpackage_missing_both(self) -> None:
        exc = self._check("snowflake.ml.modeling.xgboost", installed=set())
        assert exc is not None
        self.assertEqual(
            str(exc),
            "'snowflake.ml.modeling.xgboost' requires scikit-learn and xgboost, which are not "
            "installed. As of snowflake-ml-python 2.0, scikit-learn and xgboost are optional "
            "dependencies of snowflake-ml-python that are no longer installed automatically. "
            "Install the missing packages in your environment to continue using "
            "snowflake.ml.modeling.",
        )

    def test_lightgbm_subpackage_missing_only_lightgbm(self) -> None:
        # lightgbm was already an optional extra before 2.0, so the message does not blame 2.0.
        exc = self._check("snowflake.ml.modeling.lightgbm", installed={"sklearn"})
        assert exc is not None
        self.assertEqual(
            str(exc),
            "'snowflake.ml.modeling.lightgbm' requires lightgbm, which is not installed. lightgbm "
            "is an optional dependency of snowflake-ml-python that is not installed automatically. "
            "Install the missing package in your environment to continue using "
            "snowflake.ml.modeling.",
        )

    def test_lightgbm_subpackage_missing_both(self) -> None:
        # Mixed case: only scikit-learn's optional status is new in 2.0.
        exc = self._check("snowflake.ml.modeling.lightgbm", installed=set())
        assert exc is not None
        self.assertEqual(
            str(exc),
            "'snowflake.ml.modeling.lightgbm' requires lightgbm and scikit-learn, which are not "
            "installed. lightgbm and scikit-learn are optional dependencies of snowflake-ml-python "
            "that are not installed automatically (scikit-learn as of snowflake-ml-python 2.0). "
            "Install the missing packages in your environment to continue using "
            "snowflake.ml.modeling.",
        )

    def test_all_present_does_not_raise(self) -> None:
        exc = self._check("snowflake.ml.modeling.linear_model", installed={"sklearn"})
        self.assertIsNone(exc)

        exc = self._check("snowflake.ml.modeling.xgboost", installed={"sklearn", "xgboost"})
        self.assertIsNone(exc)

        exc = self._check("snowflake.ml.modeling.lightgbm", installed={"sklearn", "lightgbm"})
        self.assertIsNone(exc)


if __name__ == "__main__":
    absltest.main()
