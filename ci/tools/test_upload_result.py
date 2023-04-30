#
# Copyright (c) 2012-2023 Snowflake Computing Inc. All rights reserved.
#
import os
import shutil
import tempfile
import uuid

import numpy as np
import pandas as pd
from absl.testing.absltest import TestCase, main

import snowflake.connector
from ci.tools.upload_result import ResultUploader
from snowflake.ml.utils.connection_params import SnowflakeLoginOptions


class TestResultUploader(TestCase):
    _INPUT_AND_EXPECTED_OUTPUT = [
        (
            """
            [{"package": "ci/tools", "filename": "parse_coverage.py", "line_found": 64, "line_hit": 57},
            {"package": "snowflake/ml/_internal", "filename": "env.py", "line_found": 6, "line_hit": 6}]
            """.strip().replace(
                " ", ""
            ),
            pd.DataFrame(
                data={
                    "GIT_REVISION": ["test_upload_result_1" for _ in range(2)],
                    "PACKAGE": ["ci/tools", "snowflake/ml/_internal"],
                    "FILENAME": ["parse_coverage.py", "env.py"],
                    "TOTAL_LINES": [64, 6],
                    "COVERED_LINES": [57, 6],
                    "DURATION_SEC": [-1, -1],
                },
                dtype=np.int8,
            ),
        ),
        (
            """
            [{"package": "ci/tools", "filename": "parse_coverage.py", "line_found": 64, "line_hit": 57}]
            """.strip().replace(
                " ", ""
            ),
            pd.DataFrame(
                data={
                    "GIT_REVISION": ["test_upload_result_2"],
                    "PACKAGE": ["ci/tools"],
                    "FILENAME": ["parse_coverage.py"],
                    "TOTAL_LINES": [64],
                    "COVERED_LINES": [57],
                    "DURATION_SEC": [-1],
                },
                dtype=np.int8,
            ),
        ),
    ]
    conn_cfg = SnowflakeLoginOptions()
    cur = snowflake.connector.connect(**conn_cfg).cursor()

    db = conn_cfg["database"]
    schema = conn_cfg["schema"]
    temp_table1 = "temp" + str(uuid.uuid1()).replace("-", "")
    temp_table2 = "temp" + str(uuid.uuid1()).replace("-", "")
    temp_table1_full_path = f"{db}.{schema}.{temp_table1}"
    temp_table2_full_path = f"{db}.{schema}.{temp_table2}"

    def _create_tmp_table_if_not_exists(table_full_path: str, cur: snowflake.connector.cursor.SnowflakeCursor) -> None:
        """Create a temp table"""
        query = f"""
            CREATE TEMPORARY TABLE {table_full_path} (
                time TIMESTAMP_LTZ,
                git_revision VARCHAR(40) PRIMARY KEY,
                package VARCHAR,
                filename VARCHAR,
                covered_lines INT,
                total_lines INT,
                duration_sec INT
            );"""
        cur.execute(query)

    @classmethod
    def setUpClass(cls) -> None:
        cls._create_tmp_table_if_not_exists(table_full_path=cls.temp_table1_full_path, cur=cls.cur)
        cls._create_tmp_table_if_not_exists(table_full_path=cls.temp_table2_full_path, cur=cls.cur)

    def setUp(self) -> None:
        # Create temporary LCOV file for testing
        self._temp_dir_1 = tempfile.mkdtemp()
        self._temp_dir_2 = tempfile.mkdtemp()
        lcov_file_path = os.path.join(self._temp_dir_1, "breakdown_coverage.json")
        with open(lcov_file_path, "w") as lcov_file:
            lcov_file.write(self._INPUT_AND_EXPECTED_OUTPUT[0][0])
        lcov_file_path = os.path.join(self._temp_dir_2, "breakdown_coverage.json")
        with open(lcov_file_path, "w") as lcov_file:
            lcov_file.write(self._INPUT_AND_EXPECTED_OUTPUT[1][0])

    def tearDown(self) -> None:
        # Delete temporary LCOV folder
        shutil.rmtree(self._temp_dir_1)
        shutil.rmtree(self._temp_dir_2)
        # After closing the session, the temporary tables will be purged
        self.cur.close()

    def _make_uploader(self, breakdown_coverage_file_path: str, git_revision_number: str, table: str) -> None:
        result_uploader = ResultUploader(
            breakdown_coverage_file_path,
            git_revision_number=git_revision_number,
        )
        result_uploader._cursor = self.cur
        result_uploader._write_to_table(db=self.db, schema=self.schema, table=table)

    def testInvalidDirectory(self) -> None:
        _params = "not_exist_directory"

        with self.assertRaises(ValueError):
            ResultUploader(_params)

    def testInvalidTableName(self) -> None:
        table_name = "not_exist_table"
        with self.assertRaises(snowflake.connector.errors.ProgrammingError):
            result_uploader = ResultUploader(os.path.join(self._temp_dir_1, "breakdown_coverage.json"))
            result_uploader._conn_cfg = self.conn_cfg
            result_uploader._connect()
            result_uploader._write_to_table(db=self.db, schema=self.schema, table=table_name)

    # Test that uploading results at two revisions will cause rows from both revisions to be recorded.
    def testUploadTwoRevisions(self) -> None:
        git_revision_number1 = "test_upload_result_1"
        git_revision_number2 = "test_upload_result_2"

        # Write to table (first query)
        self._make_uploader(
            os.path.join(self._temp_dir_1, "breakdown_coverage.json"), git_revision_number1, self.temp_table2
        )

        # Write to table (second query)
        self._make_uploader(
            os.path.join(self._temp_dir_2, "breakdown_coverage.json"), git_revision_number2, self.temp_table2
        )

        sql = f"select * from {self.db}.{self.schema}.{self.temp_table2} where git_revision='{git_revision_number1}';"
        self.cur.execute(sql)
        sql_df_result: pd.DataFrame = self.cur.fetch_pandas_all().sort_values(by=["FILENAME"]).reset_index(drop=True)
        sql_df_expected: pd.DataFrame = (
            self._INPUT_AND_EXPECTED_OUTPUT[0][1].sort_values(by=["FILENAME"]).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sql_df_result[sql_df_expected.columns], sql_df_expected)

        sql2 = f"select * from {self.db}.{self.schema}.{self.temp_table2} where git_revision='{git_revision_number2}';"
        self.cur.execute(sql2)
        sql_df_result2: pd.DataFrame = self.cur.fetch_pandas_all().sort_values(by=["FILENAME"]).reset_index(drop=True)
        sql_df_expected2: pd.DataFrame = (
            self._INPUT_AND_EXPECTED_OUTPUT[1][1].sort_values(by=["FILENAME"]).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sql_df_result2[sql_df_expected2.columns], sql_df_expected2)

    # Test that uploading the same result (against the same git revision) twice would not cause
    def testUploadSameResultTwice(self) -> None:
        git_revision_number1 = "test_upload_result_1"

        # Write to table (first query)
        self._make_uploader(
            os.path.join(self._temp_dir_1, "breakdown_coverage.json"), git_revision_number1, self.temp_table2
        )
        # Write to table (second query)
        self._make_uploader(
            os.path.join(self._temp_dir_1, "breakdown_coverage.json"), git_revision_number1, self.temp_table2
        )

        sql = f"select * from {self.db}.{self.schema}.{self.temp_table2} where git_revision='{git_revision_number1}';"
        self.cur.execute(sql)
        sql_df_result: pd.DataFrame = self.cur.fetch_pandas_all().sort_values(by=["FILENAME"]).reset_index(drop=True)
        sql_df_expected: pd.DataFrame = (
            self._INPUT_AND_EXPECTED_OUTPUT[0][1].sort_values(by=["FILENAME"]).reset_index(drop=True)
        )
        pd.testing.assert_frame_equal(sql_df_result[sql_df_expected.columns], sql_df_expected)


if __name__ == "__main__":
    main()
