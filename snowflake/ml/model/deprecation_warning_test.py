import sys
import warnings
from collections import namedtuple
from unittest import mock

from absl.testing import absltest


class DeprecationWarningTest(absltest.TestCase):
    def test_python_39_deprecation_warning(self) -> None:
        """Test that Python 3.9 triggers deprecation warning on import."""
        # Create a version_info namedtuple that matches sys.version_info structure
        VersionInfo = namedtuple("VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"])
        mock_version = VersionInfo(3, 9, 0, "final", 0)

        with (
            mock.patch.object(sys, "version_info", mock_version),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            # Clear the module from sys.modules to force reimport
            module_name = "snowflake.ml.model"
            if module_name in sys.modules:
                del sys.modules[module_name]

            # Import the module to trigger the warning
            import snowflake.ml.model  # noqa: F401

            # Check that exactly one Python 3.9 warning was issued
            python39_warnings = [
                warning
                for warning in w
                if "Python 3.9 is deprecated in snowflake-ml-python. Please upgrade to Python 3.10 or greater."
                in str(warning.message)
                and issubclass(warning.category, DeprecationWarning)
            ]
            self.assertEqual(
                len(python39_warnings),
                1,
                f"Expected 1 warning, got {len(python39_warnings)}. All warnings: "
                f"{[str(warning.message) for warning in w]}",
            )

    def test_python_310_no_deprecation_warning(self) -> None:
        """Test that Python 3.10 does not trigger deprecation warning."""
        # Create a version_info namedtuple that matches sys.version_info structure
        VersionInfo = namedtuple("VersionInfo", ["major", "minor", "micro", "releaselevel", "serial"])
        mock_version = VersionInfo(3, 10, 0, "final", 0)

        with (
            mock.patch.object(sys, "version_info", mock_version),
            warnings.catch_warnings(record=True) as w,
        ):
            warnings.simplefilter("always")

            # Import the module
            import snowflake.ml.model  # noqa: F401

            # Check that no Python 3.9 warning was issued
            python39_warnings = [
                warning
                for warning in w
                if "Python 3.9 is deprecated in snowflake-ml-python. Please upgrade to Python 3.10 or greater."
                in str(warning.message)
            ]
            self.assertEqual(len(python39_warnings), 0)


if __name__ == "__main__":
    absltest.main()
