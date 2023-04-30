#
# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.
#

import json
import os
import shutil
import tempfile
from typing import Dict, List, Tuple, Union

from absl.testing.absltest import TestCase, main

from ci.tools.parse_coverage import LcovCoverageParser


class TestLcovParser(TestCase):
    _VALID_FILE: str = """
    SF:dir1/test_file.py
    LH:5
    LF:10
    end_of_record
    SF:dir2/dir3/another_file.py
    LH:10
    LF:20
    end_of_record
    """.strip().replace(
        " ", ""
    )

    _INVALID_FILE_WITHOUT_LF: str = """
    SF:test_file.py
    LH:5
    end_of_record
    """.strip().replace(
        " ", ""
    )

    _INVALID_FILE_WITHOUT_LH: str = """
    SF:test_file.py
    LF:20
    end_of_record
    """.strip().replace(
        " ", ""
    )

    _INVALID_FILE_WITHOUT_END: str = """
    SF:another_file.py
    LH:10
    LF:20
    """.strip().replace(
        " ", ""
    )

    _INVALID_FILE_WITH_RANDOM_ORDER: str = """
    SF:another_file.py
    LF:20
    LH:10
    end_of_record
    """.strip().replace(
        " ", ""
    )

    _INVALID_FILE_LIST: List[str] = [
        _INVALID_FILE_WITHOUT_LF,
        _INVALID_FILE_WITHOUT_LH,
        _INVALID_FILE_WITHOUT_END,
        _INVALID_FILE_WITH_RANDOM_ORDER,
    ]

    _VALID_INPUT_AND_EXPECTED_OUTPUT: List[Tuple[str, List[Dict[str, Union[str, int]]]]] = [
        (
            _VALID_FILE,
            [
                {"package": "dir1", "filename": "test_file.py", "line_found": 10, "line_hit": 5},
                {"package": "dir2/dir3", "filename": "another_file.py", "line_found": 20, "line_hit": 10},
            ],
        )
    ]

    def setUp(self) -> None:
        # Create temporary LCOV file for testing
        self._temp_dir = tempfile.mkdtemp()

        for i, invalid_lcov in enumerate(TestLcovParser._INVALID_FILE_LIST):
            lcov_file_path = os.path.join(self._temp_dir, f"invalid_test{i}.dat")
            with open(lcov_file_path, "w") as lcov_file:
                lcov_file.write(invalid_lcov + "\n")

        for i, lcov_file_to_write in enumerate(TestLcovParser._VALID_INPUT_AND_EXPECTED_OUTPUT):
            lcov_file_path = os.path.join(self._temp_dir, f"valid_test{i}.dat")
            with open(lcov_file_path, "w") as lcov_file:
                lcov_file.write(lcov_file_to_write[0] + "\n")

    def tearDown(self) -> None:
        # Delete temporary LCOV folder
        shutil.rmtree(self._temp_dir)

    def test_valid_lcov_to_json(self) -> None:
        for i in range(len(TestLcovParser._VALID_INPUT_AND_EXPECTED_OUTPUT)):
            lcov_file_path = os.path.join(self._temp_dir, f"valid_test{i}.dat")

            parse_coverage = LcovCoverageParser(lcov_file_path, self._temp_dir)
            parse_coverage.process()

            breakdown_json_path = os.path.join(self._temp_dir, "breakdown_coverage.json")

            # Open JSON file and assert contents
            with open(breakdown_json_path) as json_file:
                data = json.load(json_file)
                self.assertEqual(
                    data,
                    TestLcovParser._VALID_INPUT_AND_EXPECTED_OUTPUT[i][1],
                )

    def test_invalid_parse_lcov_to_json(self) -> None:
        # Call method to parse LCOV file to JSON
        for i in range(len(TestLcovParser._INVALID_FILE_LIST)):
            lcov_file_path = os.path.join(self._temp_dir, f"invalid_test{i}.dat")

            parse_coverage = LcovCoverageParser(lcov_file_path, self._temp_dir)

            with self.assertRaises(ValueError):
                parse_coverage.process()


if __name__ == "__main__":
    main()
