import importlib
import os
import tempfile

from absl.testing import absltest

from snowflake.ml._internal.file_utils import zip_file_or_directory_to_stream
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions
from snowflake.snowpark import Session
from snowflake.snowpark.functions import sproc
from tests.integ.snowflake.ml._internal import snowpark_handlers_test


class SnowparkHandlersTest(absltest.TestCase):
    def test_in_sproc(self) -> None:

        session = Session.builder.configs(SnowflakeLoginOptions()).create()

        # Create a temp stage to load the test module
        stage_name = "test"
        stage_creation_query = f"CREATE OR REPLACE TEMPORARY STAGE {stage_name};"
        session.sql(stage_creation_query).collect()

        tmpdir = tempfile.mkdtemp()
        zip_file_name = "snowflake-ml.zip"
        tmparchive = os.path.join(tmpdir, zip_file_name)
        snowml_path = list(importlib.import_module("snowflake.ml").__path__)[-1]

        start_path = os.path.abspath(os.path.join(snowml_path, os.pardir, os.pardir))

        zip_module_filename = os.path.join(tmpdir, zip_file_name)
        with zip_file_or_directory_to_stream(snowml_path, start_path) as input_stream:
            with open(zip_module_filename, "wb") as f:
                f.write(input_stream.getbuffer())

        # Upload the package to the stage.
        session.file.put(
            f"{tmparchive}",
            f"@{stage_name}",
            auto_compress=False,
            overwrite=True,
        )
        # Upload the test to the stage.
        session.file.put(
            f"{snowpark_handlers_test.__file__}",
            f"@{stage_name}",
            auto_compress=False,
            overwrite=True,
        )
        imports = [f"@{row.name}" for row in session.sql(f"LIST @{stage_name}").collect()]

        @sproc(
            is_permanent=False,
            packages=[
                "snowflake-snowpark-python",
                "numpy",
                "scikit-learn",
                "cloudpickle",
                "pyarrow",
                "fastparquet",
                "inflection",
                "pytest",
                "absl-py",
            ],
            replace=True,
            session=session,
            anonymous=True,
            imports=imports,
        )
        def test_sproc(session: Session) -> None:
            import sys
            import unittest
            import zipfile

            import snowpark_handlers_test

            import_dir = sys._xoptions.get("snowflake_import_directory")
            module_zip_path = import_dir + f"{zip_file_name}"
            extracted = "/tmp/snowml"

            with zipfile.ZipFile(module_zip_path, "r") as myzip:
                myzip.extractall(extracted)
            sys.path.append(extracted)

            suite = unittest.TestLoader().loadTestsFromModule(snowpark_handlers_test)
            result = unittest.TextTestRunner(verbosity=2, failfast=False).run(suite)
            if len(result.errors) > 0 or len(result.failures) > 0:
                raise RuntimeError("Unit test failed unexpectedly with at least one error.")

        test_sproc(session=session)


if __name__ == "__main__":
    absltest.main()
