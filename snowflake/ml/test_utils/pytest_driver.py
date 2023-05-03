"""Driver for running pytest tests with `bazel test`.

Example your_test.py:

from snowflake.ml.utils import pytest_driver

def test_case():
    assert some_feature()

if __name__ == "__main__":
    pytest_driver.main()

"""

import os
import sys

import pytest

# If run by bazel, JUnit XML output should be produced at the given path.
# See https://bazel.build/reference/test-encyclopedia#initial-conditions
_BAZEL_XML_OUTPUT_ENV = "XML_OUTPUT_FILE"


def main() -> None:
    args = list(sys.argv)
    xml_output_path = os.environ.get(_BAZEL_XML_OUTPUT_ENV, None)
    if xml_output_path is not None:
        args.append(f"--junitxml={xml_output_path}")
    sys.exit(pytest.main(args))
